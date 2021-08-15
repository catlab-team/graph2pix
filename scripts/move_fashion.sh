for subdir in * ; do mv $subdir/file.txt $subdir.txt; done;

for subdir in * ; do cp $subdir/ou_pp.jpg ../pix2pixHD/datasets/fashion/train_A/"$subdir"_ou_pp.jpg; done;


for subdir in * ; do echo ../pix2pixHD/datasets/fashion/train_A/"$subdir"_in_pp.jpg done;