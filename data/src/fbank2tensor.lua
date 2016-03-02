require 'torch'

function write2tensor(dest)
	local nutt = #features
	local nframes = #features[#features]	-- last utterence is the longest
	local featdim = #features[1][1]
	local featTensor  = torch.Tensor(nframes, nutt, featdim)
	local labelTensor = torch.ones(nframes, nutt)
	local uttlen      = torch.zeros(nutt)
	for i=1,nutt do
		for j=1,#features[i] do	--feature[i] consists of all the frames of utterence i
			-- feature --
			for k=1,featdim do
				featTensor[j][i][k] = features[i][j][k]
			end
			-- label --
			labelTensor[j][i] = labels[i][j]
		end
		uttlen[i] = #features[i]
	end
	local obj = {input = featTensor, target = labelTensor, uttLen = uttlen, vocab_size = #vocab}
	torch.save(dest, obj)
end

function getFeature()
	local file = assert(io.open("tmpfeature", 'r'))
	local line = file:read()
	local featTab = {}
	local curFrame
	while (line) do
		if (line:sub(1,1)~='-') then
			for k in line:gmatch("%S*") do
				if (#k > 0) then
					if (k:sub(-1,-1)==':') then
						curFrame = tonumber(k:sub(1,-2))+1
						featTab[curFrame] = {} 
					else
						featTab[curFrame][#featTab[curFrame]+1] = tonumber(k)
					end
				end
			end
		end
		line = file:read()
	end
	file:close()
	return featTab
end

function getFile(file_name)		-- Splited MLF File
	local f = assert(io.open("../mlf/" .. file_name, 'r'))
	local line = f:read()
	local curname
	local featTab, labelTab, ctclabelTab
	repeat
		if (string.len(line)==1) then
			while (#featTab ~= #labelTab) do
				print(string.format("-- Warning: In %s, the number of frames %d is different from that of labels %d --", file_name, #featTab, #labelTab))
				if (#labelTab == 0) then
					os.exit()
				end
				while (#labelTab > #featTab) do
					table.remove(labelTab)
				end
				while (#labelTab < #featTab) do
					labelTab[#labelTab+1] = labelTab[#labelTab]
				end
			end
			features[#features+1] = featTab
			labels[#labels+1] 	  = labelTab
			ctclabels[#ctclabesl+1] = ctclabelTab
			if (#features > chunksize) then	-- greater than 10240 files been read
				dest = string.format('../tensors/tensor%03d', curfile); curfile = curfile + 1
				if (curfile>startfile) then
					write2tensor(dest)
				end
				features = {}; labels = {}; ctclabels = {}
				collectgarbage()
			end
		elseif (string.sub(line, -4, -2)=='lab') then
			curname = string.sub(line, 4, -5)
			print(file_name, line, curname .. "fbank")
			if (curfile>=48) then
				chunksize = 2560
			end
			if (curfile >= startfile) then
				os.execute("/speechlab/tools/HTK/htkbin/htk64/HList ../fbank/" .. curname .. "fbank > tmpfeature")
				featTab = getFeature()
			else
				featTab = {}
			end
			labelTab = {}; ctclabelTab = {}
		else
			if (curfile>=startfile) then
				-- print(line)
				local cnt = 0
				local startt,endt,startf,endf
				for k in line:gmatch("%S*") do
					if (#k>0) then
						if (cnt==0) then
							startt = tonumber(k)
							cnt = cnt+1
						elseif (cnt==1) then
							endt = tonumber(k)-100000
							cnt = cnt+1
						else
							lb = vocab[k] 
						end
					end
				end
				startf = math.floor(startt / 200000)
				endf   = math.floor(endt   / 200000)
				for i=startf+1, endf+1 do
					labelTab[i] = lb
				end
			end
		end

		line = f:read()
	until not line
	f:close()
	collectgarbage()

end

-- Reading Vocabulary --
print("Reading Vocabulary")
local file = assert(io.open("../vocab/vocab", 'r'))
local word = file:read()
vocab = {}
local cnt = 0
while (word) do
	vocab[word] = cnt+1
	cnt = cnt + 1
	word = file:read()
end
file:close()

curfile = 1
-- Reading Features --
local fileLists = io.popen('ls ../mlf'):read("*all")
chunksize = 10240

features = {}; labels = {}; ctclabels = {}
-- write the features and labels to files in tensor format --
if (not path.exists('../tensors')) then
	if (not path.mkdir('../tensors')) then
		print("-- Error: cannot create directory ../tensors/ to store tensors --")
	end
end

startfile = 1

-- getFile("sortedmlftmp")
for k in fileLists:gmatch("sortedmlf%S+") do
	getFile(k)
end
if (#features>0) then
	local dest = string.format('../tensors/tensor%03d', curfile); curfile = curfile + 1
	write2tensor(dest)
end
