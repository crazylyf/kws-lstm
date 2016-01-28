require 'torch'
require 'nn'
require 'nngraph'
require 'optim'

local LSTM = require 'LSTM'
--local SplitMinibatchLoader = require 'SplitMinibatchLoader'
local SplitMinibatchLoader = require 'SplitMinibatchLoaderTensor'
local model_utils = require 'model_utils'
require 'misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a QbE Keyword Spotting Model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-train_data_dir', './data/tensors/train_tensors/', 'directory of the training data')
cmd:option('-eval_data_dir', './data/tensors/eval_tensors/', 'directory of the evaluating data')
cmd:option('-featscp', 'feats.scp', 'filename of input feature scp')
-- model params
cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'lstm, gru or rnn')
-- optimization
cmd:option('-learning_rate', 2e-3, 'learning rate')
cmd:option('-learning_rate_decay', 0.97, 'learning rate decay')
cmd:option('-learning_rate_decay_after', 10, 'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate', 0.95, 'decay rate for rmsprop')
cmd:option('-dropout', 0, 'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-batch_size', 128, 'number of sequences to train on in parallel')
cmd:option('-seq_length', 160, 'number of timesteps to unroll for')
cmd:option('-max_epochs', 1, 'number of full passes through the training data')
cmd:option('-grad_clip', 5, 'clip gradients at this value')
cmd:option('-train_frac', 0.95, 'fraction of data that goes into train set')
cmd:option('-val_frac', 0.05, 'fraction of data that goes into validation set')
            -- test_frac will be computed as (1 - train_frac - val_frac)
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 123, 'torch manual random number generator seed')
cmd:option('-print_every', 1, 'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every', 1000, 'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile', 'lstm', 'filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-accurate_gpu_timing', 0, 'set this flag to 1 to get precise timings when using GPU. Might make code bit slower but reports accurate timings.')
-- GPU/CPU
cmd:option('-useGPU', 0, 'Using GPU or not. 1 = use GPU, 0=use CPU')
cmd:text()

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

local test_frac = math.max(0, 1-(opt.train_frac+opt.val_frac))
local split_sizes = {opt.train_frac, opt_val_frac, test_frac}

if opt.useGPU > 0 then
	local ok, cunn = pcall(require, 'cunn')
	local ok2, cutorch = pcall(require, 'cutorch')
	if not ok then print('package cunn not found!') end
	if not ok2 then print('package cutorch no found!') end
	if ok and ok2 then
		print('using CUDA on GPU ...')
		cutorch.manualSeed(opt.seed)
	end
end

--data
-- create the data loader class
print('preparing data')
local train_loader = SplitMinibatchLoader.create(opt.train_data_dir, opt.batch_size, opt.seq_length, split_sizes)
local eval_loader = SplitMinibatchLoader.create(opt.eval_data_dir, opt.batch_size, opt.seq_length, split_sizes)
local feats_dim = train_loader.feats_dim
local vocab_size = train_loader.vocab_size
print('keywords size: ' .. vocab_size)

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
	print('loading a model from checkpoint ' .. opt.init_from)
	local checkpoint = torch.load(opt.init_from)
	protos = checkpoint.protos
	
	print('overwriting rnn_size =' .. checkpoint.opt.rnn_size .. ', num_layers = ' .. checkpoint.opt.num_layers .. ', model = ' .. checkpoint.opt.model .. ' based on the checkpoint.')
	opt.rnn_size = checkpoint.opt.rnn_size
	opt.num_layers = checkpoint.opt.num_layers
	opt.model = checkpoint.opt.model
	do_random_init = false
else
	print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
	protos = {}
	if opt.model == 'lstm' then
		protos.rnn = LSTM.lstm(feats_dim, opt.rnn_size, vocab_size, opt.num_layers, opt.dropout)
	end
	protos.criterion = nn.ClassNLLCriterion()
end

-- ship the model to the GPU is desired
if opt.useGPU > 0 then
	for k,v in pairs(protos) do v:cuda() end
end

init_state = {}
for L = 1, opt.num_layers do
	local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
	if opt.useGPU > 0 then h_init = h_init:cuda() end
	table.insert(init_state, h_init:clone())
	if opt.model == 'lstm' then
		table.insert(init_state, h_init:clone())	-- h[t-1] && c[t-1]
	end
end

--put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initalization
if do_random_init then
	params:uniform(-0.08, 0.08) -- small uniform numbers
end
-- initialize the LSTM forget gates with slightly higher biases to encourage remembering in the beginning
if opt.model == 'lstm' then
	for layer_idx = 1, opt.num_layers do
		for _,node in ipairs(protos.rnn.forwardnodes) do
			if node.data.annotations.name == 'i2h_' .. layer_idx then
				print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
				-- the gates are, in order , i,f,o,g, so f is the 2nd block of weights
				node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
			end
		end
	end
end

print('number of parameters in the model: ' .. params:nElement())

-- evaluate the loss over an entire split
function eval_split(nbatch)
	print('evaluating loss over validation set')
	
	local loss = 0
	local rnn_state=init_state
	local seq_length

	for i = 1,nbatch do -- iterate over batches in the split
		-- fetch a batch
		local inputs, targets, uttLen
		inputs, targets, uttLen, seq_length = eval_loader:next_batch()
		if opt.useGPU > 0 then
			-- have to convert to float because integers can't be cuda()'d
			inputs = inputs:float():cuda()
			targets = targets:float():cuda()
		end

		-- forward pass
		protos.rnn:evaluate() -- for dropout proper functioning
		for t = 1, seq_length do
			local lst = protos.rnn:forward{inputs[t], unpack(rnn_state)}
			rnn_state = {}
			for i=1,#init_state do table.insert(rnn_state,lst[i]) end
			predictions = lst[#lst]
			loss = loss + protos.criterion:forward(predictions,targets[t])
		end
		
		print(i .. '/' .. nbatch .. '...')
	end
	
	loss = loss / seq_length / nbatch
	return loss
end

-- do fwd/bwd and return loss, grad_parms
local init_state_global = clone_list(init_state)
function feval(x)
	if x ~= params then
		params:copy(x)
	end
	grad_params:zero()
	local timer = torch.Timer()
	-- get minibatch
	local inputs, targets, uttLen, seq_length = train_loader:next_batch()
	
	if opt.useGPU > 0 then
		-- have to convert to float because integers can't be cuda()'d
		inputs = inputs:float():cuda()
		targets = targets:float():cuda()
		uttLen = uttLen:float():cuda()
	end
	-- print(inputs:size(), targets:size())
	local time1 = timer:time().real
	-- forward pass
	local rnn_state = {[0] = init_state_global}
	local predictions = {}     -- softmax outputs
	local loss = 0
	
	-- forward pass
	print("-- forward pass --")
	protos.rnn:training()
	for t = 1, seq_length do
		local lst = protos.rnn:forward{inputs[t], unpack(rnn_state[t-1])}
		rnn_state[t] = {}
		for i = 1, #init_state do table.insert(rnn_state[t], lst[i]) end	-- extract the state, without output
		predictions[t] = lst[#lst]	-- last element is the prediction
		loss = loss + protos.criterion:forward(predictions[t], targets[t])
	end
	loss = loss / seq_length 

	local time2 = timer:time().real
	
	-- backward pass
	print("-- backward pass --")
	local drnn_state = {[seq_length] = clone_list(init_state, true)}
	for t = seq_length,1,-1 do
		local doutput_t = protos.criterion:backward(predictions[t], targets[t])
		for i = 1, opt.batch_size do
			if uttLen[i]<t then
				doutput_t[i]:zero()
			end
		end
		table.insert(drnn_state[t], doutput_t)
		local dlst = protos.rnn:backward({input[t], unpack(rnn_state[t-1])}, drnn_state[t])
		drnn_state[t-1] = {}
		for k,v in pairs(dlst) do
			if k > 1 then	-- k==1 is gradient on x, which we don't need
				-- note we do k-1 because first item is dembeddings, and then follow the
				-- derivatives of the state, starting at index 2.
				drnn_state[t-1][k-1] = v
			end
		end
	end

	-- init_state_global = rnn_state
	init_stage_global = clone_list(init_state)
	grad_params:clamp(-opt.grad_clip, opt.grad_clip)
	local time3 = timer:time().real
	print(string.format("loadingTime / batch = %.4fs, forwardTime / batch = %.4fs, backwardTime / batch = %.4fs", time1, time2, time3))
	return loss,grad_params
end

-- start optimization here
-- train_losses = {}
-- val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
--local iterations = opt.max_epochs * train_loader.nbatches
--local iterations_per_epoch = train_loader.nsamples
local loss0 = nil
local i=0
local epoch=1
local lastsample=1
local eval
repeat
	--local epoch = i / train_loader.nbatches
	i = i+1
	if (train_loader.curSample==1 and lastsample ~= 1) then
		epoch = epoch + 1
		eval = true
	end
	lastsample = train_loader.curSample

	local timer = torch.Timer()
	local _, loss = optim.rmsprop(feval, params, optim_state)
	if opt.useGPU > 0 then
		cutorch.synchronize()
	end
	
	local time = timer:time().real

	local train_loss = loss[1]
	-- train_losses[i] = train_loss

	--if i % iterations_per_epoch == 0 and opt.learning_rate_decay < 1 then
	if opt.learning_rate_decay < 1 then
		if epoch >= opt.learning_rate_decay_after then
			local decay_factor = opt.learning_rate_decay
			optim_state.learningRate = optim_state.learningRate * decay_factor
			print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
		end
	end

	--if i % opt.eval_val_every == 0 or i == iterations then
	if eval or i % opt.eval_val_every == 0 then
		-- evaluate loss on validation data
		local nbatch = eval_loader.nbatches
		local val_loss = eval_split(nbatch) 
		eval_loader.curidx = 1	-- reset the start index of evaluation data to 1
		-- val_losses[i] = val_loss
		eval = false
		print(string.format("%d(epoch %d), eval_loss = %6.8f", i, epoch, val_loss))
		
		local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
		print('saving checkpoint to ' .. savefile)
		local checkpoint = {}
		checkpoint.protos = protos
		checkpoint.opt = opt
		--checkpoint.train_losses = train_losses
		checkpoint.val_loss = val_loss
		--checkpoint.val_losses = val_losses
		checkpoint.i = i
		checkpoint.epoch = epoch
		torch.save(savefile, checkpoint)
	end

	if i % opt.print_every == 0 then
		--print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
		print(string.format("%d(epoch %d), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, epoch, train_loss, grad_params:norm() / params:norm(), time))
	end

	if i % 10 == 0 then collectgarbage() end
	
	if loss[1] ~= loss[1] then
		print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
		break -- halt
	end

	if loss0 == nil then loss0 = loss[1] end
	if loss[1] > loss0 * 3 then
		print('loss is exploding, aborting.')
		break
	end
until epoch > 5 
