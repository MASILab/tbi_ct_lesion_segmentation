require(ichseg)
DATA_DIR <- "~/tbi_ct_lesion_segmentation/data/test"
TARGET_DIR <- "~/tbi_ct_lesion_segmentation/data/test/ich_segs"

filenames <- Filter(function(x) {grepl("_CT.nii.gz", x)}, list.files(DATA_DIR))
filepaths <- lapply(filenames, (function(x) file.path(DATA_DIR, x)))
target_filepaths <- lapply(filenames, (function(x) file.path(TARGET_DIR, x)))

filepaths <- unlist(filepaths)
target_filepaths <- unlist(target_filepaths)

for (i in 1:length(filepaths)) {
    ich_segment(img=filepaths[i], outfile=target_filepaths[i])
}

