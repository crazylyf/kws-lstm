local LSTM = {}
function LSTM.lstm(input_size, rnn_size, output_size, nlayers, dropout)
	dropout = dropout or 0

	local inputs = {}
	table.insert(inputs, nn.Identity()()) --x
	for L = 1,nlayers do
		table.insert(inputs, nn.Identity()()) --prev_c[L]
		table.insert(inputs, nn.Identity()()) --prev_h[L]
	end

	local x, input_size_L
	local outputs = {}
	for L = 1,nlayers do
		-- c,h from previous timesteps
		local prev_c = inputs[L*2]
		local prev_h = inputs[L*2+1]
		-- the input to this layer
		if L == 1 then
			x = inputs[1]
			input_size_L = input_size
		else
			x = outputs[(L-1)*2]
			if dropout > 0 then x = nn.Dropout(dropout)(x) end
			input_size_L = rnn_size
		end
		--evaluate the input sums at once for efficiency
		local i2h = nn.Linear(input_size_L, 4*rnn_size)(x):annotate{name='i2h_'..L}
		local h2h = nn.Linear(rnn_size, 4*rnn_size)(prev_h):annotate{name='h2h_'..L}
		local all_input_sums = nn.CAddTable()({i2h, h2h})

		local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
		local n1,n2,n3,n4 = nn.SplitTable(2)(reshaped):split(4)

		local in_gate = nn.Sigmoid()(n1)
		local forget_gate = nn.Sigmoid()(n2)
		local out_gate = nn.Sigmoid()(n3)

		local in_transform = nn.Tanh()(n4)
		local next_c = nn.CAddTable()({
			  nn.CMulTable()({forget_gate, prev_c}),
			  nn.CMulTable()({in_gate, in_transform})
			})

		local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

		table.insert(outputs, next_c)
		table.insert(outputs, next_h)
	end

	local top_h = outputs[#outputs]
	if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
	local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{nn='decoder'}
	local logsoft = nn.LogSoftMax()(proj)
	table.insert(outputs, logsoft)
	return nn.gModule(inputs, outputs)
end

return LSTM
