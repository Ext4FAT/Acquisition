#include "Dataset.hpp"
#include "DrawWorld.hpp"
#include "Macro.hpp"
#include "Opencv.hpp"
#include "Realsense.hpp"
#include "HOG-SVM.hpp"
//thread
#include <thread>
using std::thread;

void myimshow(const string winname, Mat &img)
{
	imshow(winname, img);
}

void threadShow(const string winname, Mat &img)
{
	thread t(myimshow, winname, img);
	t.detach();
}


//Show boundbox and word
inline void drawText(Mat &img, Rect &boundbox, const string content)
{
	putText(img, content, (boundbox.tl() + boundbox.br()) / 2, 3, 0.6, Scalar(0, 0, 255), 2);
}

// Locate windows position
void placeWindows(int topk)
{
	cv::namedWindow("depth");
	cv::namedWindow("color");
	cv::namedWindow("before merging");
	cv::namedWindow("segmentation");
	cv::namedWindow("classification");
	cv::namedWindow("regions");
	cv::moveWindow("depth", 0, 0);
	cv::moveWindow("color", 350, 0);
	cv::moveWindow("segmentation", 1050, 0);
	cv::moveWindow("before-merging", 700, 0);
	cv::moveWindow("classification", 350, 300);
	cv::moveWindow("regions", 0, 300);
	for (int k = 0; k < topk; k++) {
		cv::namedWindow(to_string(k));
		cv::moveWindow(to_string(k), (k + 2) * 350, 300);
	}
}

//Dir example: "..\\saveData\\"
Dataset::Dataset(const string& Dir, int width, int height, float fps /*= 60*/) :
dir_(Dir), depthDir_(Dir + "depth\\"), colorDir_(Dir + "color\\"), pcdDir_(Dir + "pcd\\"), fps_(fps)
{
	camera_.width = width;
	camera_.height = height;
}

//Convert RealSense's PXCImage to Opencv's Mat
Mat Dataset::PXCImage2Mat(PXCImage* pxc)
{
	if (!pxc)	return Mat(0, 0, 0);
	PXCImage::ImageInfo info = pxc->QueryInfo();
	PXCImage::ImageData data;
	Mat cvt;
	if (info.format & PXCImage::PIXEL_FORMAT_YUY2) {	//color data
		if (pxc->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_RGB24, &data) < PXC_STATUS_NO_ERROR)
			return  Mat(0, 0, 0);
		cvt = Mat(info.height, info.width, CV_8UC3, (void*)data.planes[0],data.pitches[0]/sizeof(uchar));
	}
	else if (info.format & PXCImage::PIXEL_FORMAT_DEPTH) {//depth data
		if (pxc->AcquireAccess(PXCImage::ACCESS_READ, PXCImage::PIXEL_FORMAT_DEPTH, &data) < PXC_STATUS_NO_ERROR)
			return  Mat(0, 0, 0);
		cvt = Mat(info.height, info.width, CV_16U, (void*)data.planes[0], data.pitches[0] / sizeof(uchar));	//Mat初始化是按照长宽来定的
	}
	pxc->ReleaseAccess(&data);
	return cvt;
}

//Save PXC Point Cloud to PCD file
int Dataset::savePCD(const string& outfilename, Segmentation &myseg)
{
	ofstream ofs(outfilename);
	ofs << "# .PCD v0.7 - Point Cloud Data file format" << endl;
	ofs << "VERSION 0.7" << endl;
	ofs << "FIELDS x y z" << endl;
	ofs << "SIZE 4 4 4" << endl;
	ofs << "TYPE F F F" << endl;
	ofs << "COUNT 1 1 1" << endl;
	ofs << "WIDTH " << myseg.mainRegions_[0].size() << endl;
	ofs << "HEIGHT 1" << endl;
	ofs << "VIEWPOINT 0 0 0 1 0 0 0" << endl;
	ofs << "POINTS " << myseg.mainRegions_[0].size() << endl;
	ofs << "DATA ascii" << endl;
	//double scale = 1. / 300;
	double scale = 1. / 330;
	//vector<PXCPoint3DF32> obj_cloud;
	for (auto& p : myseg.mainRegions_[0]) {
		p += p;
		PXCPoint3DF32 ppp = vertices_[p.y * 640 + p.x];
		ofs << ppp.x*scale << " " << ppp.y*scale << " " << ppp.z*scale << endl;
		//obj_cloud.push_back(vertices[p.y * 640 + p.x]);
	}
	ofs.close();
	return 0;
}

//Acquire color/depth/pcd data
int Dataset::dataAcquire()
{
	// Define variable
	Mat color, depth, display;
	vector<PXCPoint3DF32> vertices(camera_.height*camera_.width);
	PXCSession *pxcsession;
	PXCSenseManager *pxcsm;
	PXCCapture::Device *pxcdev;
	PXCProjection *projection;
	PXCCapture::Sample *sample;
	PXCImage *pxcdepth,*pxccolor;
	long framecnt;
	// Configure RealSense
	pxcsession = PXCSession::CreateInstance();
	pxcsm = pxcsession->CreateSenseManager();
	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, camera_.width, camera_.height, fps_);
	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, camera_.width, camera_.height, fps_);
	pxcsm->Init();
	pxcdev = pxcsm->QueryCaptureManager()->QueryDevice();
	pxcdev->SetDepthConfidenceThreshold(4);
	if (!pxcdev) {
		MESSAGE_COUT("ERROR", "Failed to create an SDK SenseManager");
		return -1;
	}
	projection = pxcdev->CreateProjection();
	if (!projection) {
		MESSAGE_COUT("ERROR", "Failed to create an SDK Projection");
		return -2;
	}
	// Configure Point Cloud Show
	DrawWorld dw(pxcsession, camera_);
	PXCPoint3DF32 light = { .5, .5, 1.0 };
	// Configure Segmentation
	unsigned topk = 5;
	short threshold = 3;
	Segmentation myseg(320, 240, topk, threshold);
	// Configure Classification
	HOG_SVM classification(".\\gwj-dataset\\HOG-SVM-MODEL.xml");
	vector<string> names = classification.getSubdirName(".\\gwj-dataset");
	classification.getCategory(names);
	//
	placeWindows(0);
	// Detect each video frame
	for (framecnt = 1; (1); ++framecnt) {
		if (pxcsm->AcquireFrame(true) < PXC_STATUS_NO_ERROR)	break;
		// Query the realsense color and depth, and project depth to color
		try{
			sample = pxcsm->QuerySample();
			pxcdepth = sample->depth;
			pxccolor = sample->color;
			pxcdepth = projection->CreateDepthImageMappedToColor(pxcdepth, pxccolor);
			// Generate and Show 3D Point Cloud

			//pxcStatus sts = projection->QueryVertices(pxcdepth, &vertices[0]);
			//if (sts >= PXC_STATUS_NO_ERROR) {
			//	PXCImage* drawVertices = dw.DepthToWorldByQueryVertices(vertices, pxcdepth, light);
			//	if (drawVertices){
			//		Mat display = PXCImage2Mat(drawVertices);
			//		imshow("display", display);
			//	}
			//}

			//pxcdepth = projection->CreateColorImageMappedToDepth(pxcdepth,pxccolor);
			depth = PXCImage2Mat(pxcdepth);
			color = PXCImage2Mat(pxccolor);
			if (!depth.cols || !color.cols)	continue;

			Mat depth2, color2;
			resize(depth, depth2, Size(320, 240));
			resize(color, color2, Size(320, 240));

			threadShow("color", color2);

			//imshow("depth", 65535 / 1200 * depth2);
			//imshow("color", color2);

			myseg.Segment(depth2, color2);
			
			// save regions
			if (' ' == waitKey(1)) {
				time_t now = time(0);
				int count = 0;
				for (auto &boundbox : myseg.boundBoxes_) {
					string filename = getSaveFileName(now, 100 * framecnt + ++count);
					string cpath = colorDir_ + filename;
					string dpath = depthDir_ + filename;
					imwrite(cpath, color2(boundbox));
					imwrite(dpath, depth2(boundbox));
				}
				MESSAGE_COUT("[" << framecnt << "]", "write OK");
			}


			// show hog_svm
			{
				Mat classify = color2.clone();
				time_t now = time(0);
				int count = 0;
				for (auto &boundbox : myseg.boundBoxes_) {
					Mat reg = color2(boundbox);
					int predict = classification.predict(reg);
					if (predict > 0) {
						cout << predict << endl;
						Category c = classification.category();
						rectangle(classify, boundbox, Scalar(0, 0, 255), 2);
						drawText(classify, boundbox, c.getName(predict));
						//string filename = getSaveFileName(now, 100 * framecnt + ++count);
						//string cpath = colorDir_ + filename;
						//string dpath = depthDir_ + filename;
						//imwrite(cpath, color2(boundbox));
						//imwrite(dpath, depth2(boundbox));
					}
				}
				threadShow("classification", classify);
				//imshow("classification", classify);
			}
			// Clear Segmentation data; 
			myseg.clear();
			// Release Realsense SDK memory and read next frame 
			pxcdepth->Release();
			pxcsm->ReleaseFrame();

		}
		catch (cv::Exception e){
			MESSAGE_COUT("ERROR", e.what());
		}
	}
	return 1;

}


int Dataset::show()
{
	/************************************************************************/
	/* Define variable                                                      */
	/************************************************************************/
	Mat color, depth, display;
	PXCSession *pxcsession;
	PXCSenseManager *pxcsm;
	PXCCapture::Device *pxcdev;
	PXCProjection *projection;
	PXCCapture::Sample *sample;
	PXCImage *pxcdepth, *pxccolor;
	long framecnt;
	/************************************************************************/
	/* Configure RealSense                                                  */
	/************************************************************************/
	pxcsession = PXCSession::CreateInstance();
	pxcsm = pxcsession->CreateSenseManager();
	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, camera_.width, camera_.height, fps_);
	pxcsm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, camera_.width, camera_.height, fps_);
	//Configure Draw
	PXCPoint3DF32 light = { .5, .5, 1.0 };
	DrawWorld dw(pxcsession, camera_);
	//Configure Segmentation
	unsigned topk = 5;
	short threshold = 3;
	Segmentation myseg(320, 240, topk, threshold);
	//placeWindows(topk);
	
	
	//Query Information
	pxcsm->Init();
	pxcdev = pxcsm->QueryCaptureManager()->QueryDevice();

	//PXCPointF32 cf = pxcdev->QueryColorFocalLength();
	//PXCPointF32 df = pxcdev->QueryDepthFocalLength();
	//pxcF32 cfmm = pxcdev->QueryColorFocalLengthMM();
	//pxcF32 dfmm = pxcdev->QueryDepthFocalLengthMM();
	//PXCPointF32 dpp = pxcdev->QueryDepthPrincipalPoint();
	//PXCPointF32 cpp = pxcdev->QueryColorPrincipalPoint();
	//PXCPointF32 dview = pxcdev->QueryDepthFieldOfView();
	//PXCRangeF32 drange = pxcdev->QueryDepthSensorRange();
	//PXCCapture::DeviceInfo ppp;
	//pxcdev->QueryDeviceInfo(&ppp);
	//cout << cfmm << endl;
	//cout << dfmm << endl;
	//cout << endl;


	//setthreshold
	//pxcdev->SetDepthConfidenceThreshold(4);


	if (!pxcdev) {
		MESSAGE_COUT("ERROR", "Failed to create an SDK SenseManager");
		return -1;
	}
	projection = pxcdev->CreateProjection();
	if (!projection) {
		MESSAGE_COUT("ERROR", "Failed to create an SDK Projection");
		return -1;
	}

	//calibration
	//PXCCalibration *calib = projection->QueryInstance<PXCCalibration>();
	//PXCCalibration::StreamCalibration sc;
	//PXCCalibration::StreamTransform st;
	//calib->QueryStreamProjectionParametersEx(PXCCapture::StreamType::STREAM_TYPE_DEPTH,
	//	PXCCapture::Device::StreamOption::STREAM_OPTION_DEPTH_PRECALCULATE_UVMAP,
	//	&sc, &st);
	//cout << endl;

	/************************************************************************/
	/* Detect each video frame                                              */
	/************************************************************************/
	vertices_.resize(camera_.height*camera_.width);
	time_t now;
	for (	framecnt = 1, now = time(0);
			-1 == waitKey(1); 
			++framecnt, now = time(0), pxcdepth->Release(), pxcsm->ReleaseFrame()	) 
	{
		if (pxcsm->AcquireFrame(true) < PXC_STATUS_NO_ERROR)	break;
		//Query the realsense color and depth, and project depth to color
		sample = pxcsm->QuerySample();
		pxcdepth = sample->depth;
		pxccolor = sample->color;
		pxcdepth = projection->CreateDepthImageMappedToColor(pxcdepth, pxccolor);

		pxcStatus sts = projection->QueryVertices(pxcdepth, &vertices_[0]);
		if (sts >= PXC_STATUS_NO_ERROR) {
			PXCImage* drawVertices = dw.DepthToWorldByQueryVertices(vertices_, pxcdepth, light);
			if (drawVertices){
				display = PXCImage2Mat(drawVertices);
				//imshow("display", display);
			}
		}

		depth = PXCImage2Mat(pxcdepth);
		color = PXCImage2Mat(pxccolor);
		if (!depth.cols || !color.cols)	continue;

		Mat depth2, color2;
		resize(depth, depth2, Size(320, 240));
		resize(color, color2, Size(320, 240));

		Mat show = color2.clone();

		//myseg.completeDepth(depth);

		imshow("depth", 65535 / 1200 * depth2);
		//imshow("color", color2);

		myseg.Segment(depth2, color2);
		resize(display, display, Size(320, 240));


		int S = 1;
		//myseg.mainRegions_.size();
		for (int k = 0; k < S; k++){
			for (auto p : myseg.mainRegions_[k]) {
				show.at<Vec3b>(p) = display.at<Vec3b>(p);
			}
			myimshow(to_string(k), show);
			//imshow(to_string(k), show);
		}


		//vector<PXCPoint3DF32> obj_cloud;
		//static Point dir[5] = { { 0, 0 }, { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };
		//static Rect range(0, 0, 640, 480);
		//for (int k = 0; k < S; k++){
		//	Mat show = Mat::zeros(Size(640, 480), CV_8U);

		//	for (auto p : myseg.mainRegions_[k]) {
		//		cout << p.x <<" "<< p.y << endl;
		//		for (auto d : dir) {
		//			Point pp = p * 2 + d;
		//			if (!pp.inside(range)) continue;
		//			cout << p.x << " " << p.y << endl;
		//			obj_cloud.push_back(vertices[pp.y * 640 + pp.x]);
		//			show.at<Vec3b>(pp) = display.at<Vec3b>(pp);
		//		}
		//	}
		//	imshow(to_string(k), show);
		//}


		//myseg.Segment(depth, color);

		

		if (waitKey(1) == ' ') {
			//string filename = getSaveFileName(now, framecnt);
			//string pcdFilepath = pcdDir_ + filename + ".pcd";
			//string rgbFilepath = colorDir_ + filename;
			//string depthFilepath = depthDir_ + filename;
			//{//record regions;
			//	string recordPath = dir_ + filename + ".txt";
			//	ofstream record(recordPath);
			//	for (auto &seg : myseg.mainRegions_) {
			//	}
			//	record.close();
			//}
			//string filename = "realsense.pcd";
			string filename = getSaveFileName(now, framecnt) + ".pcd";
			cout << filename << endl;
			
			savePCD(filename, myseg);

			cout << "OK" << endl;


 			waitKey(-1);
		}
		//Release
		myseg.clear();
	}
	return 1;
}

int Dataset::testSVM(string dir) 
{
	Size showSize = { camera_.width / 2, camera_.height / 2 };
	//Configure Segmentation
	unsigned topk = 5;
	short threshold = 3;
	Segmentation myseg(showSize.width, showSize.height, topk, threshold);
	//Configure HOG-SVM
	HOG_SVM hog_svm(".\\classification\\HOG-SVM-MODEL.xml");
	//
	//vector<string> categories = { "can", "teacup", "teapot", "box"};
	vector<string> categories = { "bottle" };
	//
	for (auto categoryname : categories) {
		vector<string> filenames = getCurdirFileName(dir + "\\" + categoryname);
		for (auto filename : filenames) {
			string color_path = dir + "\\" + categoryname + "\\" + filename;
			string depth_path = dir + "\\depth\\" + filename;
			//Segment by depth data
			Mat depth, color, depth2, color2;
			color = imread(color_path, CV_LOAD_IMAGE_UNCHANGED);
			depth = imread(depth_path, CV_LOAD_IMAGE_UNCHANGED);
			resize(depth, depth2, showSize);
			resize(color, color2, showSize);
			clock_t start = clock();
			myseg.Segment(depth2, color2);
			cout << "[Segmentation]\t" << 1.0*(clock() - start) / CLOCKS_PER_SEC << endl;
			int count = 0;
			for (auto &boundbox : myseg.boundBoxes_) {
				start = clock();
				Mat region = color2(boundbox);
				stringstream ss;
				ss << filename << "-" << boundbox << ".png";
				if (!count++) {
					imwrite(".\\dataset\\" + categoryname + "\\1\\" + ss.str(), region);
				}
				else {
					imwrite(".\\dataset\\" + categoryname + "\\-1\\" + ss.str(), region);
				}
			}


			//int count = 0;
			//for (auto &boundbox : myseg.boundBoxes_) {
			//	start = clock();
			//	Mat region = color2(boundbox);
			//	int predict = hog_svm.predict(region);
			//	++count;
			//	if (predict == 1) {
			//		string name = to_string(predict);
			//		//rectangle(color2, boundbox, Scalar(0, 0, 255), 2);
			//		//drawText(color2, boundbox, name);
			//		cout << "[" << filename << "-" << count << "]\t" << 1.0*(clock() - start) / CLOCKS_PER_SEC << endl;
			//		imwrite(".//1//" + filename + "-" + to_string(count) + ".png", region);
			//	}
			//	else {
			//		imwrite(".//-1//" + filename + "-" + to_string(count) + ".png", region);
			//	}
			//}
			//start = clock();
			
			cout << "[imwrite]\t" << 1.0*(clock() - start) / CLOCKS_PER_SEC << endl;
			myseg.clear();
		}
	}
	return 1;
}


int Dataset::testRegion(string dir = ".\\dataset")
{
	//Configure HOG-SVM
	HOG_SVM hog_svm(".\\classification\\HOG-SVM-MODEL.xml");
	
	//
	vector<string> categories = {"background", "bottle", "box", "can", "teacup", "teapot" };
	map<string, int> reverse_categories;
	for (int i = 0; i < categories.size(); i++) {
		reverse_categories[categories[i]] = 1+i;
	}
	//reverse_categories["background"] = -1;
	//
	for (auto categoryname : categories) {
		vector<string> filenames = getCurdirFileName(dir + "\\" + categoryname);
		int failed = 0;
		for (auto filename : filenames) {
			string path = dir + "\\" + categoryname + "\\" + filename;
			Mat region = imread(path, CV_LOAD_IMAGE_UNCHANGED);
			int predict = hog_svm.predict(region);
			//cout << predict << endl;
			if (predict != reverse_categories[categoryname]) {
				failed++;
			}
		}
		MESSAGE_COUT(categoryname, failed << " / " << filenames.size());
	}
	return 1;
}
