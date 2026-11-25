-- Trusted caller supplies policy; one Redis key per identity/rule.
-- ARGV: capacity, refill tokens/sec, cost. Returns allowed, remaining, retry_ms.
local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
if capacity <= 0 or rate <= 0 or cost <= 0 or cost > capacity then
  return redis.error_reply('invalid policy or cost')
end
local clock = redis.call('TIME')
local now = tonumber(clock[1]) + tonumber(clock[2]) / 1000000
local row = redis.call('HMGET', KEYS[1], 'tokens', 'updated')
local tokens = tonumber(row[1]) or capacity
local updated = tonumber(row[2]) or now
now = math.max(now, updated) -- do not mint tokens from a backward wall-clock step
-- Clamp balance when a policy lowers capacity.
tokens = math.min(capacity, tokens + (now-updated) * rate)
local allowed = 0
local retry = 0
if tokens >= cost then
  tokens = tokens-cost
  allowed = 1
else
  retry = math.ceil((cost-tokens)/rate*1000)
end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'updated', now)
redis.call('PEXPIRE', KEYS[1], math.ceil(capacity/rate*1000)+1000)
return {allowed, math.floor(tokens), retry}
