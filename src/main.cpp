#include "Dataset.hpp"
#include "HOG-SVM.hpp"

int main(int argc, char** argv)
{
	//string save_dir_path = ".\\savedData\\";
	//Dataset data(save_dir_path, 640, 480, 30);
	//data.dataAcquire();

	HOG_SVM hog_svm;
	hog_svm.EndToEnd(".\\ZYY-DESKTOP-ITEMS\\");
	system("pause");
	return 1;
}



