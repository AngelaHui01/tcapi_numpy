# tests/tcapi_numpy_test.py
import tcapi
import numpy as np


Ten = tcapi.TenKindF64
ctx = tcapi.create_context(Ten)


# --- version ---
print("TCAPI version:", tcapi.version(Ten))  # 1.0


# --- show ---
A = tcapi.fill(ctx, Ten, (2, 3), 5.0)
tcapi.show(ctx, A)


# --- save & load ---
tcapi.save(ctx, A, "test_tensor")
A_loaded = tcapi.load(ctx, Ten, "test_tensor")
print("load dtype match:", A_loaded.dtype == Ten.dtype)
print("load values match:", np.array_equal(A, A_loaded))


# --- load dtype mismatch ---
try:
    tcapi.load(ctx, tcapi.TenKindF32, "test_tensor")
except TypeError as e:
    print(f"Caught expected error: {e}")


# --- close ---
B = tcapi.fill(ctx, Ten, (2, 3), 5.0)
C = tcapi.fill(ctx, Ten, (2, 3), 5.001)

print("close (eps=0.01):", tcapi.close(ctx, A, C, 0.01))
print("close (eps=1e-6):", tcapi.close(ctx, A, C, 1e-6))
print("close exact:     ", tcapi.close(ctx, A, B, 0.0))

try:
    tcapi.close(ctx, A, B, -1.0)
except ValueError as e:
    print(f"Caught expected error: {e}")


# --- convert ---
# A NumPy backend context may support multiple TenKinds, so ctx can be both
# the source and destination context.
A_f32 = tcapi.convert(ctx, A, ctx, tcapi.TenKindF32)
print("convert f64->f32 dtype:", A_f32.dtype)

A_c128 = tcapi.convert(ctx, A, ctx, tcapi.TenKindC128)
print("convert f64->c128 dtype:", A_c128.dtype)

# Same TenKind must create a deep copy.
A_copy = tcapi.convert(ctx, A, ctx, Ten)
print("same-kind copy dtype:", A_copy.dtype)
print("same-kind copy values:", np.array_equal(A, A_copy))
print("same-kind copy separate memory:", not np.shares_memory(A, A_copy))


tcapi.destroy_context(ctx)
print("done:", ctx)