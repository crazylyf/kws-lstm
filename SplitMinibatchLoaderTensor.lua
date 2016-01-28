local SplitMinibatchLoader = {}
SplitMinibatchLoader.__index = SplitMinibatchLoader

function SplitMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions)
	-- split_fractions is e.g. {0.9, 0.05, 0.05}
	local self = {}
	setmetatable(self, SplitMinibatchLoader)

	self.nsamples = 0	-- there are nsamples tensor files to be read
	local fileLists = io.popen('ls ' .. data_dir):read("*all")
	for k in fileLists:gmatch('%S*tensor%S+') do
		self.nsamples = self.nsamples + 1
	end
	
	self.curSample = 1

	self.obj = torch.load(data_dir .. string.format("tensor%03d",self.curSample))
	--self.vocab_size = self.obj.vocab_size
	self.vocab_size = 0
	local file = io.open('./data/vocab/vocab','r')
	local line = file:read()
	while (line) do
		if (#line>0) then
			self.vocab_size = self.vocab_size + 1
		end
		line = file:read()
	end
	self.curidx = 1	-- current tensor been read
	self.feats_dim = self.obj.input:size(3)
	self.batch_size = batch_size
	self.data_dir = data_dir
	self.nbatches = math.floor(self.obj.input:size(2) / batch_size)

	print(string.format('data load done'))
	collectgarbage()
	return self
end

function SplitMinibatchLoader:next_batch()
	-- print("-- loading the new batch --")
	local seq_length = self.obj.input:size(1)
	-- prepare to load next tensor file --
	local tmpobj
	if (self.curidx + self.batch_size-1 > self.obj.input:size(2)) then
		self.curSample = self.curSample + 1
		if self.curSample > self.nsamples then
			self.curSample = 1
		end
		tmpobj = torch.load(self.data_dir .. string.format("tensor%03d", self.curSample))
		if seq_length < tmpobj.input:size(1) then
			seq_length = tmpobj.input:size(1)
		end
	end
	local input = torch.zeros(seq_length, self.batch_size, self.feats_dim)
	local target = torch.ones(seq_length, self.batch_size):long()
	local uttLen = torch.ones(self.batch_size):long()
	
	local j=1
	while (self.curidx <= self.obj.input:size(2) and j<=self.batch_size) do
		uttLen[j] = self.obj.uttLen[self.curidx]
		for i = 1, self.obj.uttLen[self.curidx] do	-- utterence length dimension
			input[i][j] = self.obj.input[i][self.curidx]
			target[i][j] = self.obj.target[i][self.curidx]
		end
		j = j + 1
		self.curidx = self.curidx + 1
	end
	if (j<=self.batch_size) then
		self.obj = tmpobj
		self.curidx = 1
		while (self.curidx <= self.obj.input:size(2) and j<=self.batch_size) do
			uttLen[j] = self.obj.uttLen[self.curidx]
			for i = 1, self.obj.uttLen[self.curidx] do
				input[i][j] = self.obj.input[i][self.curidx]
				target[i][j] = self.obj.target[i][self.curidx]
			end
			j = j+1
			self.curidx= self.curidx
		end
	end	

	return input, target, uttLen, seq_length	-- curNum is used to record the number of meaningful frames, the rest of frames in the input/target are random tensors.
end

return SplitMinibatchLoader
