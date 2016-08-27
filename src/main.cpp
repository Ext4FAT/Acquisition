#include "Dataset.hpp"
#include "HOG-SVM.hpp"

int main(int argc, char** argv)
{
	//HOG_SVM hog_svm;
	////hog_svm.BinaryClassification(".\\dataset\\bottle", ".\\dataset\\Background");
	//hog_svm.EndToEnd(".\\dataset\\");
	
	
	//string save_dir_path = ".\\savedData\\";
	//Dataset data(save_dir_path, 640, 480, 30);
	//data.dataAcquire();


	string save_dir_path = ".\\savedData\\";
	Dataset data(save_dir_path, 640, 480, 30);
	//data.testSVM("C:\\Users\\IDLER\\Desktop\\DATASET\\DataSet");
	data.testRegion(".\\dataset");


	system("pause");

	return 1;
}



