function g_reset_state(state)
  state.gpos = 1
  if model ~= nil and model.gstart_s ~= nil then
    for d = 1, 2 * params.layers do
      model.gstart_s[d]:zero()
    end
  end
end

function g_reset_ds()
  for d = 1, #model.gds do
    model.gds[d]:zero()
  end
end
  
function fpG(state)
  g_replace_table(model.gs[0], model.gstart_s)
  state.gpos = state.pre_pos
  --print('G pos '..state.gpos)
  if state.gpos + params.seq_length > state.data:size(1) then
    g_reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.gpos]
    local y = state.data[state.gpos + 1]
    local s = model.gs[i - 1]
    _, model.gs[i] = unpack(model.grnns[i]:forward({x, y, s}))
    state.gpos = state.gpos + 1
  end
  g_replace_table(model.gstart_s, model.gs[params.seq_length])
  return -1
end

function bpG(state)
  gparamdx:zero()
  g_reset_ds()
  for i = params.seq_length, 1, -1 do
    state.gpos = state.gpos - 1
    local x = state.data[state.gpos]
    local y = state.data[state.gpos + 1]
    local s = model.gs[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.grnns[i]:backward({x, y, s},
                                       {derr, model.gds})[3]
    g_replace_table(model.gds, tmp)
    cutorch.synchronize()
  end
  state.gpos = state.gpos + params.seq_length
  model.gnorm_dw = gparamdx:norm()
  if model.gnorm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.gnorm_dw
    gparamdx:mul(shrink_factor)
  end
end
-- Update functions
function update_reset_state(state)
  --state.gpos = 1
  if model ~= nil and model.update_start_s ~= nil then
    for d = 1, 2 * params.layers do
      model.update_start_s[d]:zero()
    end
  end
end

function fp_update(state)
  g_replace_table(model.gs[0], model.update_start_s)
  if state.gpos + params.seq_length > state.data:size(1) then
    state.gpos = 1
    update_reset_state(state)
  end
  for i = 1, params.seq_length do
    local x = state.data[state.gpos]
    local y = state.data[state.gpos + 1]
    local s = model.gs[i - 1]
    _, model.gs[i] = unpack(model.grnns[i]:forward({x, y, s}))
    state.gpos = state.gpos + 1
  end
  g_replace_table(model.update_start_s, model.gs[params.seq_length])
  return -1
end

function update_status(state)
  status:zero()
  update_reset_state(state)
  state.gpos = torch.random(state.data:size(1))
  for i = 1, opt.gBatch do
    local e = fp_update(state)
    bpG(state)
    status:add(gparamdx)
  end
  status:div(opt.gBatch)
end
