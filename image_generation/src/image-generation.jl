using DataFrames


function fill_brain(roi_locs::DataFrame)
  df_right = by(roi_locs, :roi) do df
    roi = string("R", df[:roi][1])
    DataFrame(x=df[:x], y=df[:y], z=df[:z], tmp=roi)
  end
  df_left = by(roi_locs, :roi) do df
    roi = string("L", df[:roi][1])
    DataFrame(x=-1*df[:x], y=df[:y], z=df[:z], tmp=roi)
  end
  ret = vcat(df_left, df_right)
  delete!(ret, :roi)
  rename!(ret, :tmp, :roi)
end


function map_roi_vals(data_col::Symbol,
                      roi_mets::DataFrame = readtable("../../analysis/output/local.csv"),
                      roi_locations_path::ASCIIString="../data/cluster_rois_4D.txt")
  roi_locs::DataFrame = begin
    df = readtable(roi_locations_path, names=[:x, :y, :z, :roi], separator=' ')
    ret = fill_brain(df)
  end

  ret = by(roi_locs, :roi) do df
    roi::ASCIIString = df[:roi][1]
    roi_val::Float64 = roi_mets[roi_mets[:node] .== roi, data_col][1]
    DataFrame(x = df[:x], y=df[:y], z=df[:z], roi_val=roi_val)
  end

  begin
    delete!(ret, :roi)
  end

  dest_path = string("../output/", data_col, ".txt")
  writetable(dest_path, ret, separator=' ', header=false)

  ret
end
