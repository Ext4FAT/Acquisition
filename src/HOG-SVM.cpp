
#include "HOG-SVM.hpp"
#include "Macro.hpp"

HOG_SVM::HOG_SVM()
{
    svm = SVM::create();
}

HOG_SVM::HOG_SVM(std::string model_path)
{
    svm = Algorithm::load<SVM>(model_path);
}

bool HOG_SVM::LoadModel(std::string model_path)
{
    bool flag = true;
    try{
        svm = Algorithm::load<SVM>(model_path);
    }
    catch (std::exception e){
		MESSAGE_COUT("ERROR", e.what());
        flag = false;
    }
    return flag;
}

Mat HOG_SVM::ExtractFeature(Mat Img, Size mrs)
{
    /**
     * @brief HOG_SVM::ExtractFeature
        The story behind 1764
        For example
        window size is 64x64, block size is 16x16 and block setp is 8x8£¬cell size is 8x8,
        the block number window contained is (£¨64-16£©/8+1)*((64-16)/8+1) = 7*7 = 49,
        the cell number each block contained is (16/8)*(16/8) = 4
        every cell can project 9 bin, and each bin related to 9 vector
        so feature_dim  = B x C x N, and caulated result is  1764
        (B is each window's blocks number, C is every block's cell number, n is bin number)
     */
    resize(Img, Img, mrs);
    HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
    std::vector<float> descriptors;
    hog->compute(Img, descriptors, Size(1, 1), Size(0, 0));
    return Mat(descriptors).t();
}

Mat HOG_SVM::GetDataSet(std::vector<std::string> data_path)
{
    int nImgNum = data_path.size();
    int success = 0;
    Mat data_mat, src;
    for (int i = 0; i < nImgNum; i++){
        src = imread(data_path[i]);
        if (src.cols && src.rows){
			MESSAGE_COUT("PROCESS", FileOperation::findFileName(data_path[i]) << "\t" << success++);
            Mat post = ExtractFeature(src, Size(64, 64));
            data_mat.push_back(post);
        }
    }
    return data_mat;
}

Mat HOG_SVM::GetDataSet(std::vector<std::string> data_path, std::vector<GroundTruth>& gt, int c)
{
    int nImgNum = data_path.size();
    int success = 0;
    Mat data_mat;	//feature matrix
    Mat src;
    std::string imgname;
    for (int i = 0; i < nImgNum; i++){
        src = imread(data_path[i]);
        if (src.cols && src.rows){
            imgname = FileOperation::findFileName(data_path[i]);
			MESSAGE_COUT("PROCESS", imgname << "\t" << success++);
            Mat post = ExtractFeature(src, Size(64, 64));
            data_mat.push_back(post);
            gt.push_back(GroundTruth(c, imgname));
        }
    }
    return data_mat;
}

int HOG_SVM::SetSvmParameter(int sv_num, int c_r_type, int kernel, double gamma)
{
    TermCriteria criteria = TermCriteria(CV_TERMCRIT_EPS, sv_num, FLT_EPSILON);	//max support vectocr 200
    svm->setType(c_r_type);
    svm->setKernel(kernel);
    if (kernel == SVM::RBF)	svm->setGamma(gamma);
    svm->setTermCriteria(criteria);
    return 1;
}

int HOG_SVM::Training(Mat& trainSet, Mat& label, bool save,std::string dir)
{
    SetSvmParameter(200, SVM::C_SVC, SVM::LINEAR, 0);
    Ptr<TrainData> traindata = cv::ml::TrainData::create(trainSet, ROW_SAMPLE, label);
    svm->train(traindata);
    if (save){
		svm->save(dir + "HOG-SVM-MODEL.xml");
    }
    return 1;
}

int HOG_SVM::Testing(Mat& testSet, float gt)
{
    int error = 0;
    int postnum = testSet.rows;
    Mat res = Mat::zeros(postnum, 1, CV_32FC1);
    svm->predict(testSet, res);
    for (int i = 0; i < postnum; i++)
        if (res.at<float>(i, 0) != gt)
            error++;
    std::cout << error << "/" << postnum << std::endl;
    return error;
}

int HOG_SVM::Testing(Mat& testSet, std::vector<GroundTruth> gt)
{
    int error = 0;
    int postnum = testSet.rows;
    Mat res = Mat::zeros(postnum, 1, CV_32FC1);
    svm->predict(testSet, res);
    for (int i = 0; i < postnum; i++)
        if (res.at<float>(i, 0) != gt[i].label){
			MESSAGE_COUT("ERROR", gt[i].imgname << "\t" << gt[i].label);
            error++;
        }
	MESSAGE_COUT("RESULT", error << "/" << postnum);
    return error;
}

float HOG_SVM::Predict(Mat& image)
{
    if (!image.rows)	return	-1;
    Mat gray;
    cvtColor(image, gray, CV_BGR2GRAY);
    Mat post = ExtractFeature(gray, Size(64, 64));
    gray.release();
    return svm->predict(post);
}

float HOG_SVM::EndToEnd(std::string data_path)
{

	std::cout << svm->getDefaultName() << endl;
	return 0;
    //Testset path
    //std::string testPath = data_path + "test\\";
    //std::vector<std::string> testPathPositive = FileOperation::getCurrentDir(testPath + "1\\");
    //std::vector<std::string> testPathNegative = FileOperation::getCurrentDir(testPath + "-1\\");
	//Trainset path

	//std::vector<std::string> trainPathPositive = getCurdirFilePath(data_path + "bottle\\");
	//std::vector<std::string> trainPathNegative = getCurdirFilePath(data_path + "Background\\");
 //   //Get trainset
 //   Mat trainSet, label;
 //   Mat trainSetP = GetDataSet(trainPathPositive);
 //   Mat labelP = Mat::ones(trainSetP.rows, 1, CV_32SC1);
 //   Mat trainSetN = GetDataSet(trainPathNegative);
 //   Mat labelN = Mat::ones(trainSetN.rows, 1, CV_32SC1)*(-1);
 //   trainSet.push_back(trainSetP);
 //   trainSet.push_back(trainSetN);
 //   label.push_back(labelP);
 //   label.push_back(labelN);
 //   //Training model
 //   Training(trainSet, label, true, data_path);
 //   return 1.0f;


    //Testing mode
    //std::vector<GroundTruth> gtP, gtN;
    //Mat testSetP = GetDataSet(testPathPositive, gtP, 1);
    //Mat testSetN = GetDataSet(testPathNegative, gtN, -1);
    //int error = Testing(testSetP, gtP) + Testing(testSetN, gtN);
    //return 1.0f*error / (testSetP.rows + testSetN.rows);

}
