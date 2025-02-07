function getsrc(f_name)
  curr_folder = pwd()
  proj_folder = dirname(curr_folder)
  joinpath(proj_folder, "src", f_name)
end

macro include_src(f_name)
  quote
    src = getsrc($f_name)
    include(src)
  end
end
