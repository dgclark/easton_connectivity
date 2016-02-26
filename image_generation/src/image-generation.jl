using DataFrames

function map_roi_vals(roi_locations_path::ASCIIString,
                      roi_map::Dict{Int64, Float64},
                      dest_path::ASCIIString)
  roi_locs::DataFrame = begin
    df = DataFrame(readdlm(roi_locations_path, header = false))
    df[:x4] = round(Int64, df[:x4])
    df
  end

  ret = by(roi_locs, :x4) do df
    x4::Int64 = df[:x4][1]
    new_x4 = roi_map[x4]
    DataFrame(x1 = df[:x1], x2=df[:x2], x3=df[:x3], tmp=new_x4)
  end

  begin
    delete!(ret, :x4)
    rename!(ret, :tmp, :x4)
  end

  writetable(dest_path, ret, header=false, separator=' ')

  ret
end
