#include "Dataset.hpp"
#include "HOG-SVM.hpp"

int main(int argc, char** argv)
{
	HOG_SVM hog_svm;
	hog_svm.BinaryClassification(".\\IDLER-DESKTOP-ITEMS\\bottle", ".\\IDLER-DESKTOP-ITEMS\\Background");
	//hog_svm.EndToEnd(".\\IDLER-DESKTOP-ITEMS\\");
	
	
	string save_dir_path = ".\\savedData\\";
	Dataset data(save_dir_path, 640, 480, 30);
	data.dataAcquire();





	system("pause");

	return 1;
}



