using Base.Test

include("test-helpers.jl")
@include_src "connectivity_svr.jl"

roi_cols = [:L1, :L2, :R1, :R2]

multi_cols = [:L1_L1, :L1_L2, :L1_R1, :L1_R2,
              :L2_L2, :L2_R1, :L2_R2,
              :R1_R1, :R1_R2, :R2_R2]

@test diagnify(roi_cols) == multi_cols

data = [1 2 3 4
        5 6 7 8.
        9 10 11 12
        13 14 15 16]
df = DataFrame(L1=data[:, 1], L2=data[:, 2], R1=data[:, 3], R2=data[:, 4])

ts = transform_subjects(copy(df), Symbol[])

expected_ts = begin
  inner_data = [1. 2 3 4 4 6 8 9 12 16
                25 30 35 40 36 42 48 49 56 64
                81 90 99 108 100 110 120 121 132 144
                169 182 195 208 196 210 224 225 240 256]
  ret = DataFrame()
  for (i, c) in enumerate(multi_cols)
    ret[c] = inner_data[:, i]
  end
  ret
end

@test ts == expected_ts

ts2 = transform_subjects(copy(df), [:L1])

expected_ts2 = copy(expected_ts)
expected_ts2[:L1] = df[:L1]

@test ts2 == expected_ts2
