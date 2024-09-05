import math

# Proposal Specifications
dao_num = 40
proposal_num = 50
total_proposal = dao_num * proposal_num
req_per_proposal = 4
total_requests = total_proposal * req_per_proposal
proposal_tokens = 2250
proposal_output_tokens = 1000
req_time = 1

# Model Specifications
model_name = "gpt-3.5-turbo"
cost_input_tokens = 0.5
cost_output_tokens = 1.5 

# Tier Specifications
tier_tpm = 60000
tier_rpm = 3500
tier_rpd = 10000

req_allowed_per_min_speed = math.floor(60 / req_time)
req_allowed_per_min_tpm = math.floor(tier_tpm / proposal_tokens)
req_allowed_per_min = min(req_allowed_per_min_speed, req_allowed_per_min_tpm, tier_rpm)

cost_input_tokens = cost_input_tokens * total_proposal * proposal_tokens / 1000000
cost_output_tokens = cost_output_tokens * total_proposal * proposal_output_tokens / 1000000
required_days = total_requests / tier_rpd

print(f"We are using {model_name}")
print(f"The cost of input tokens is ${cost_input_tokens} and the cost of output tokens is ${cost_output_tokens}")
if (required_days < 1):
    print(f"The processing time in days and minutes is 0 days {math.ceil(total_requests / req_allowed_per_min)} minutes")
else:
    days = math.floor(required_days)
    minutes = math.ceil((total_requests - tier_rpd * required_days) / req_allowed_per_min)
    print(f"The processing time in days and minutes is {days} days {minutes} minutes")