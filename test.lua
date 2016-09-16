-- Test functionality

require 'nn'
local c = require 'trepl.colorize'
local r = c.red
local cuda
if arg[1] == '-c' then
   cuda = true
   require 'cunn'
   print('Running on ' .. c.green 'GPU')
else
   print('Running on ' .. c.blue 'CPU')
end
require 'cunn'
require 'tripletCriterion'

-- Criterion
-- [output] forward(input, target)
-- [gradInput] backward(input, target)

-- local loss = nn.TripletCriterion(samples, blocks, norm, margin)
-- samples: the number of faces sampled from each identity in a batch
-- blocks: the number of identities in a batch (samples x blocks < batchSize)
-- norm: Lp-norm for distances between embeddings (default 2)
-- margin: a hypersphere margin between anchor-positive and anchor-negative
--          pairs (default 0.2)

-- Options for specific test t
for t = 1, 4 do
   local bs = {25, 20, 25, 20}
   local em = {15, 15, 15, 15}
   local sm = { 2,  3,  5,  1}
   local bl = {10,  4,  4,  0}

   opt = {}
   opt.batchSize = bs[t]
   opt.embSize = em[t]
   opt.samples = sm[t]
   opt.blocks = bl[t]

   for a, b in pairs(opt) do print(r(a) .. ': ' .. b) end
   assert(bs[t] >= sm[t] * bl[t], r 'batchSize < samples x blocks')

   torch.manualSeed(10)
   emb = torch.randn(opt.batchSize, opt.embSize)
   local norm = nn.Normalize(2)
   emb = norm:forward(emb)
   trg = torch.Tensor(opt.batchSize)
   for i = 0, opt.blocks - 1 do
      for j = 1, opt.samples do
         trg[i*opt.samples + j] = i + 1
      end
   end
   local nbClass = bl[t] > 0 and bl[t] or math.floor(bs[t] / 4)
   trg[{ {sm[t] * bl[t] + 1, bs[t]} }]:random(nbClass)

   print(r 'emb: '); print(emb)
   print(r 'targets: '); print(trg:view(1, -1))

   crit = nn.TripletCriterion(opt.samples, opt.blocks)
   if cuda then
      emb = emb:cuda(); trg = trg:cuda()
      crit:cuda()
      loss = crit:forward(emb, trg)
      print(r 'dist: '); print(crit.dist)
      print(r 'idx: '); print(crit.embeddings + 1)
      print(r 'loss: '); print(crit.loss:view(1, -1))
      print(r 'error: ' .. loss)
      gradInput = crit:backward(emb, trg)
      print(gradInput)
   else
      loss = crit:forward(emb, trg)
      print(r 'dist: '); print(crit.dist)
      print(r 'idx: '); print(crit.embeddings)
      print(r 'loss: '); print(crit.loss:view(1, -1))
      print(r 'error: ' .. loss)
      gradInput = crit:backward(emb, trg)
      print(gradInput)
   end
end
