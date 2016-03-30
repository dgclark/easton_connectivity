using DataFrames
using Memoize
using MLBase

include("utils.jl")
include("connectivity_mediation.jl")

pre_process = mk_pre_process_fn()

function diagnify(syms::Vector{Symbol})
  num_syms::Int64 = length(syms)

  ret::Vector{Symbol} = begin
    num_multis::Int64 = round(Int64, num_syms*(num_syms + 1) / 2)
    Array(Symbol, num_multis)
  end

  ix::Int64 = 1
  for (i, s) = enumerate(syms)
    for (i2, s2) in enumerate(syms[i:end])
      ret[ix] = symbol(s, "_", s2)
      ix = ix + 1
    end
  end

  ret
end


function transform_subjects(data::DataFrame= begin
                              df = all_data
                              pre_process(df)
                              df
                            end,
                            covars::Vector{Symbol}=default_covars)
  roi_cols::Vector{Symbol} = get_roi_cols(data)

  num_rois = length(roi_cols)
  num_subj = size(data, 1)
  num_covars = length(covars)

  syms_dia = diagnify(roi_cols)
  num_syms_dia = length(syms_dia)

  ret = DataFrame(repmat([Float64], num_syms_dia),
                  syms_dia,
                  num_subj)

  ut_mask = tril(ones(Bool, num_rois, num_rois))

  for s in 1:num_subj
    multis::Array{Float64, 1} = begin
      row = DataArray(data[s, roi_cols])[:]
      mat = row * row'
      @assert size(mat) == (num_rois, num_rois)
      mat[ut_mask][:]
    end

    #hack, cant set several cols simultaneously
    for i in 1:num_syms_dia
      ret[s, i] = multis[i]
    end
  end

  for c in covars
    ret[c] = data[c]
  end

  ret

end


function load_X()
  raw_data = all_data
  covars = pre_process(raw_data)
  transform_subjects(raw_data, covars)
end


function load_Y(target::Symbol, raw_data=nothing)
  if raw_data == nothing
    raw_data = all_data
  end
  raw_data[target]
end


