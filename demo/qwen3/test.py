import torch
import mirage as mi
from collections import defaultdict
MAX_SEQ_LEN = 1
tokens = torch.full((1, MAX_SEQ_LEN), 0, dtype=torch.long)
step_tensor = torch.tensor([0], dtype=torch.int32)

dummy = torch.tensor([0])
mt = defaultdict(list)
mt["step"] = step_tensor
mt["tokens"] = tokens
mt["input_tokens"] = dummy
mt["output_tokens"] = dummy
mt["num_new_tokens"] = dummy
mt["prompt_lengths"] = dummy
mt["qo_indptr_buffer"] = dummy
mt["paged_kv_indptr_buffer"] = dummy
mt["paged_kv_indices_buffer"] = dummy
mt["paged_kv_last_page_len_buffer"] = dummy

profiler_tensor = None
mpk = mi.PersistentKernel(
    mode = "online",
    max_seq_length=MAX_SEQ_LEN,
    world_size=1,
    mpi_rank=0,
    num_workers=1,
    num_local_schedulers=1,
    num_remote_schedulers=0,
    #meta_tensors={"step":step_tensor, "tokens":tokens, "input_tokens":None, "output_tokens":None, "qo_indptr_buffer":None, "paged_kv_indptr_buffer":None, "paged_kv_indices_buffer": None, "paged_kv_last_page_len_buffer":None},
    meta_tensors = mt,
    profiler_tensor=profiler_tensor,
    max_num_batched_requests= 1,
    max_num_batched_tokens= 1,
    max_num_pages= 1,
    page_size= 1,
    eos_token_id= 0,
    trace_name= "blah",
    spec_decode_config= None,
    use_cutlass_kernel=False,
)

torch_tensor = torch.randn((1,10))

x = mpk.attach_input(torch_tensor=torch_tensor, name="x")
w = mpk.attach_input(torch_tensor=torch.randn((10,20)), name="w")
w2 = mpk.attach_input(torch_tensor=torch.randn((20,30)), name="w2")
'''w = mpk.new_tensor(
    dims=(10, 20),
    dtype=mi.bfloat16,
    name="w",
    io_category="cuda_tensor",
)'''
y = mpk.new_tensor(
    dims=(1, 20),
    dtype=mi.bfloat16,
    name="y",
    io_category="cuda_tensor",
)

z = mpk.new_tensor(
    dims=(1, 30),
    dtype=mi.bfloat16,
    name="z",
    io_category="cuda_tensor",
)

mpk.custom_kernel([x], [w], [y], (1, 1, 1), (1, 1, 1))
mpk.custom_kernel([y], [w2], [z], (1, 1, 1), (1, 1, 1))

mpk.compile(output_dir=".")
mpk()