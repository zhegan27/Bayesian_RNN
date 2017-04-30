--[[ A plain implementation of SGD

ARGS:

- `opfunc` : a function that takes a single input (X), the point
             of a evaluation, and returns f(X) and df/dX
- `x`      : the initial point
- `config` : a table with configuration parameters for the optimizer
- `config.learningRate`      : learning rate
- `config.learningRateDecay` : learning rate decay
- `config.weightDecay`       : weight decay
- `config.weightDecays`      : vector of individual weight decays
- `config.momentum`          : momentum
- `config.dampening`         : dampening for momentum
- `config.nesterov`          : enables Nesterov momentum
- `config.learningRates`     : vector of individual learning rates
- `state`  : a table describing the state of the optimizer; after each
             call the state is modified
- `state.evalCounter`        : evaluation counter (optional: 0, by default)

RETURN:
- `x`     : the new x vector
- `f(x)`  : the function, evaluated before the update

(Clement Farabet, 2012)
]]
function sgld(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local N = config.N
   local lr = config.learningRate or 1e-3
   local lrd = config.learningRateDecay or 0
   local wd = config.weightDecay or 0
   
   state.noise =state.noise or torch.Tensor():typeAs(x):resizeAs(x):zero()
   state.evalCounter = state.evalCounter or 0
   local nevals = state.evalCounter
   assert(not nesterov or (mom > 0 and damp == 0), "Nesterov momentum requires a momentum and zero dampening")

   -- (1) evaluate f(x) and df/dx
   local _ ,dfdx = opfunc(1)
   if wd~=0 then
     dfdx:add(wd, x)
   end

   -- (4) learning rate decay (annealing)
   local clr = lr / (1 + nevals*lrd)
   x:add(-clr, dfdx)
   x:add(math.sqrt(2*clr)/N, state.noise:normal())

   -- (6) update evaluation counter
   state.evalCounter = state.evalCounter + 1

   -- return x*, f(x) before optimization
   return x,{fx}
end
