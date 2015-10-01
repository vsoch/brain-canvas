# We want to generate a web interface with all pairwise voxels, meaning a Z score from one map vs a Z score from a second map, 
# rendered in the browser. We will use the render-queue-js script to place these pairs of voxels in a "queue," and
# we will attempt a strategy that let's the user clear and restart the queue for different brain regions, so it is 
# never the case of needing to wait through an entire region to come to a subsequent one. 

# When that is working, we will reassess and figure out a strategy for rendering of multiple regions at once.

# READ IN DATA, APPLY BRAIN MASK, GENERATE VECTORS

import nibabel
import numpy
from pybraincompare.mr.datasets import get_mni_atlas, get_pair_image, get_mni_atlas
from pybraincompare.compare.mrutils import get_standard_mask, make_binary_deletion_mask, resample_images_ref
from pybraincompare.compare.maths import calculate_atlas_correlation
from nilearn.masking import apply_mask

image1,image2 = get_pair_images(voxdims=["2","2"])
image1 = nibabel.load(image1)
image2 = nibabel.load(image2)
brain_mask  = nibabel.load(get_standard_mask("FSL"))
atlas = get_mni_atlas("2")["2"]
pdmask = make_binary_deletion_mask([image1,image2])

# Combine the pdmask and the brainmask
mask = numpy.logical_and(pdmask,brain_mask.get_data())
mask = nibabel.nifti1.Nifti1Image(mask,affine=brain_mask.get_affine(),header=brain_mask.get_header())

# Resample images to mask
images_resamp, ref_resamp = resample_images_ref([image1,image2],
                                                mask,
                                                interpolation="continuous")

# Apply mask to images, get vectors of Z score data
vectors = apply_mask(images_resamp,mask)

# Get atlas vectors
atlas_nii = nibabel.load(atlas.file)
atlas_vector = apply_mask([atlas_nii],mask)[0]
atlas_labels =  ['%s' %(atlas.labels[str(int(x))].label) for x in atlas_vector]
atlas_colors = ['%s' %(atlas.color_lookup[x.replace('"',"")]) for x in atlas_labels]

# Now apply the regional mask
data_table = scatterplot_compare_vector(image_vector1=vectors[0],
                                                     image_vector2=vectors[1],
                                                     image_names=["image1","image2"],
                                                     atlas_vector=atlas_vector,
                                                     atlas_labels=atlas_labels,
                                                     atlas_colors=atlas_colors)
                               
data_table_summary = calculate_atlas_correlation(image_vector1,image_vector2,atlas_vector,atlas_labels,
                                atlas_colors,corr_type="pearson",summary=True)
data_table = calculate_atlas_correlation(image_vector1,image_vector2,atlas_vector,atlas_labels,
                                atlas_colors,corr_type="pearson",summary=False)

# Write data to tsv file, first for all regions
data_table.columns = ["x","y","atlas","label","corr","color"]
data_table_summary.columns = ["labels","corr"]
data_table.to_csv("data/scatterplot_data.tsv",sep="\t",index=False)

# Write data to tsv file, now for individual regions
for region in data_table_summary["labels"]:
  subset = data_table[data_table.label==region]
  subset.to_csv("data/%s_data.tsv" %(region),sep="\t",index=False)
