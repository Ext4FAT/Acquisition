# Realsense Operation
## Acquire Depth / Color / Point Cloud from Realsense
`save_dir_path`: which dir to save data, and must contains three subdirs: `color`, `depth` and `pcd`.
The next three numbers represent resolution and FPS.
### Example
```c++
Dataset data(save_dir_path, 640, 480, 30);
data.dataAcquire();
```
## Train SVM model
class HOG_SVM has member function `EndToEnd`, which need you to provide `datapath`.
`datapath` must contains some subdirs which represent object categories, like this:
```
.\\IDLER-DESKTOP-ITEMS
.\\IDLER-DESKTOP-ITEMS\\bottle
.\\IDLER-DESKTOP-ITEMS\\teapot
.\\IDLER-DESKTOP-ITEMS\\cup
...	
```
### Example				
```c++
HOG_SVM hog_svm;
hog_svm.EndToEnd(".\\IDLER-DESKTOP-ITEMS\\");
```
## Predict with SVM model
`model_path`: a xml file which stored svm info, like this:
```xml
<?xml version="1.0"?>
<opencv_storage>
<opencv_ml_svm>
  <format>3</format>
  <svmType>C_SVC</svmType>
  <kernel>
    <type>LINEAR</type></kernel>
  <C>1.</C>
  <term_criteria><epsilon>1.1920928955078125e-007</epsilon></term_criteria>
  <var_count>1764</var_count>
  <class_count>2</class_count>
  <class_labels type_id="opencv-matrix">
    <rows>2</rows>
    <cols>1</cols>
    <dt>i</dt>
    <data>
      -1 1</data></class_labels>
  <sv_total>1</sv_total>
  <support_vectors>
    <_>
	...
	...
	...
	</_></support_vectors>
  <decision_functions>
    <_>
      <sv_count>1</sv_count>
      <rho>-3.4039191395148438e+000</rho>
      <alpha>
        1.</alpha></_></decision_functions></opencv_ml_svm>
</opencv_storage>
```
`predict` function can output the image label, which -1 represents Background
### Example
```c++
HOG_SVM hog_svm;
hog_svm.loadModel(model_path);
...
hog_svm.predict(img);
...
```