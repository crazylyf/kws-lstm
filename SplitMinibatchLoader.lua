local SplitMinibatchLoader = {}
SplitMinibatchLoader.__index = SplitMinibatchLoader

function SplitMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions, stage)
	-- split_fractions is e.g. {0.9, 0.05, 0.05}
	local self = {}
	setmetatable(self, SplitMinibatchLoader)

	local featscp_file = path.join(data_dir, stage .. '_feats.scp')
	local text_file = path.join(data_dir, 'text')
	local vocab_file = path.join(data_dir, 'vocab')

	local run_prepro = false
	if not(path.exists(vocab_file)) then
		print('vocab file do not exist. Running preprocessing...')
		run_prepro = true
	end
	
	-- get feature dimension
	if not (path.exists('feat_dim')) then
		os.execute('/slfs1/users/xkc09/TOOLS/kaldi-trunk/src/featbin/feat-to-dim scp:' .. featscp_file .. ' - > feat_dim')
	end
	self.fileReader = io.open('feat_dim', 'r')
	self.feats_dim = self.fileReader:read()
	self.fileReader:close()

	if run_prepro then
		-- construct a tensor with all the data, and vocab file
		print('one-time setup: preprocessing text file ' .. text_file .. '...')
		SplitMinibatchLoader.text_to_tensor(text, vocab_file)
	end

	print('loading vocab files')
	local obj = torch.load(vocab_file)
	self.nsamples = obj.nsamples
	self.batch_size = batch_size
	self.nbatches = obj.nfiles / batch_size
	self.vocab = obj.vocabulary
	self.seq_length = seq_length

	-- count vocab
	self.vocab_size = 0
	for _ in pairs(self.vocab) do
		self.vocab_size = self.vocab_size + 1
	end

	print('converting ark to txt readable file...')
	if not (path.exists(stage .. '_input.txt')) then
		os.execute('/slfs1/users/xkc09/TOOLS/kaldi-trunk/src/featbin/copy-feats scp:' .. featscp_file .. ' ark,t:' .. stage .. '_input.txt')
	end
	print('loading input file...')
	self.input = stage .. '_input.txt'
	self.fileReader = assert(io.open(self.input, 'r'))

	print(string.format('data load done'))
	collectgarbage()
	return self
end

function SplitMinibatchLoader:next_batch()
	print("-- loading the new batch --")
	local input = torch.Tensor(self.seq_length, self.batch_size, self.feats_dim)
	local target = torch.ones(self.seq_length, self.batch_size)
	local uttLen = torch.ones(self.batch_size)

	local curUNum = 1	-- current utterence number
	local curFNum = 0	-- current frame number
	local readnext = false
	
	repeat
		rawdata = self.fileReader:read()
		if not rawdata then
			self.fileReader:close()
			self.fileReader = io.open(self.input, 'r')
			rawdata = self.fileReader:read()
		end
		
		if (string.sub(rawdata,-1,-1) == '[') then
			if (curUNum>1) then
				uttLen[curUNum-1] = curFNum
			end
			for x in rawdata:gmatch("%S*") do
				self.curt = self.vocab[x]	-- current target label
				curFNum = 0
				break
			end
		else
			if (curFNum < self.seq_length) then
				curFNum = curFNum + 1
				target[curFNum][curUNum] = self.curt
				if (rawdata:sub(-1,-1) == ']') then
					rawdata = rawdata:sub(1,-2)
					readnext = true
				end
				local curDNum=1	-- current dimension in the current frame of current utterence
				for x in rawdata:gmatch("%S*") do
					if (string.len(x)>0) then
						input[curFNum][curUNum][curDNum] = tonumber(x)
						curDNum = curDNum + 1
					end
				end
				if (readnext) then
					curUNum = curUNum + 1
					readnext = false
				end
			end
		end

	until curUNum > self.batch_size

	return input, target, uttLen	-- curNum is used to record the number of meaningful frames, the rest of frames in the input/target are random tensors.
end

function SplitMinibatchLoader.text_to_tensor(in_textfile, out_vocab_file)
	local timer = torch.Timer()
	
	local boolVocab = {}
	-- read the text file for tags
	print('loading text file...')
	local ncnt = 0	-- number of audio samples for the input data
	local f = assert(io.open('text', 'r'))
	local rawdata
	local text_table = {}
	rawdata = f:read()
	repeat
		ncnt = ncnt + 1	-- current number of lines

		local splitlist={}
		for x in rawdata:gmatch('%S*') do
			if string.len(x)~=0 then table.insert(splitlist, x) end
		end

		text_table[splitlist[1]] = ''
		for a=2,#splitlist do
			text_table[splitlist[1]] = text_table[splitlist[1]] .. ' ' .. splitlist[a]
		end
		if not boolVocab[text_table[splitlist[1]]] then boolVocab[text_table[splitlist[1]]] = true end
		rawdata = f:read()
	until not rawdata

	f:close()
	
	local nfile = 0
	for k,v in pairs(text_table) do nfile = nfile + 1 end
	-- sort into a table
	local vocabTab = {}
	for kword in pairs(boolVocab) do vocabTab[#vocabTab+1] = kword end
	table.sort(vocabTab)
	local vocab = {}
	for i, kword in ipairs(vocabTab) do
		vocab[kword] = i
	end

	for name, kword in pairs(text_table) do
		text_table[name] = vocab[text_table[name]]
	end

	-- save vocab files
	print('saving ' .. out_vocab_file)
	local obj
	obj = {nsamples = ncnt, nfiles = nfile, vocabulary = text_table}
	torch.save(out_vocab_file, obj)

end

return SplitMinibatchLoader
