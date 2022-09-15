package com.example.hdapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import android.util.Log;
import android.view.MenuItem;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.content.pm.ActivityInfo;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.View;

// OpenCV Classes
import org.opencv.android.JavaCameraView;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfInt4;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.core.TermCriteria;

import java.util.ArrayList;
import java.util.List;



public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {

    private CameraBridgeViewBase javaCameraView;
    private int camDim[] = {320, 240}; // camera dimensions
    private float offsetFactX, offsetFactY;
    private float scaleFactX, scaleFactY;
    private boolean handDetected = false;
    private Mat ycc;
    private Scalar handColor;
    private Scalar minHSV;
    private Scalar maxHSV;
    private Mat frame, frame2;
    private Point palmCenter;
    private List<Point> fingers;
    private TermCriteria termCriteria;
    private List<Rect> allRoi;
    private List<Mat> allRoiHist;
    private MatOfFloat ranges;
    private MatOfInt channels;
    private Mat dstBackProject;
    private MatOfPoint palmContour;
    private MatOfPoint hullPoints;
    private MatOfInt hull;
    private Mat hierarchy;
    private Mat touchedMat;
    private MatOfInt4 convexityDefects;
    private Mat nonZero;
    private Mat nonZeroRow;
    private List<MatOfPoint> contours;

    private int speedTime = 0;
    private int speedFingers = 0;

    // Used for logging success or failure messages
    private static final String TAG = "OCVSample::Activity";


        // Initial check for OpenCV
    static {
        if (!OpenCVLoader.initDebug())
            Log.e("init", "OpenCV NOT loaded");
        else
            Log.e("init", "OpenCV successfully loaded");
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    javaCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // set default orientation to landscape
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        // prevent screen from going to sleep
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // holds the window's background drawable in full screen
        View decorView = getWindow().getDecorView();
        int uiOptions = View.SYSTEM_UI_FLAG_FULLSCREEN;
        decorView.setSystemUiVisibility(uiOptions);

        setContentView(R.layout.activity_main);

        // Permissions for Android 6+
        ActivityCompat.requestPermissions(MainActivity.this,
                new String[]{Manifest.permission.CAMERA},
                1);

        javaCameraView = (JavaCameraView) findViewById(R.id.java_surface_view);

        javaCameraView.setVisibility(SurfaceView.VISIBLE);

        javaCameraView.setCameraPermissionGranted();

        javaCameraView.setCvCameraViewListener(this);

        javaCameraView.setMaxFrameSize(camDim[0], camDim[1]);
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String permissions[],
                                           int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case 1: {
                // If request is cancelled, the result arrays are empty.
                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    javaCameraView.setCameraPermissionGranted();  // <------ THIS!!!
                } else {
                    // permission denied
                }
                return;
            }
        }
    }

    public void onDestroy() {
        super.onDestroy();
        if (javaCameraView != null)
            javaCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {

        // initialize global variables on camera start

        setScaleFactors(width, height);
        ycc = new Mat(height, width, CvType.CV_8UC3);
        handColor = new Scalar(255);
        minHSV = new Scalar(3);
        maxHSV = new Scalar(3);
        frame = new Mat();
        termCriteria = new TermCriteria(TermCriteria.COUNT | TermCriteria.EPS, 10, 1);
        allRoi = new ArrayList<>();
        allRoiHist = new ArrayList<>();
        ranges = new MatOfFloat(0, 180);
        channels = new MatOfInt(0);
        dstBackProject = new Mat();
        palmContour = new MatOfPoint();
        hullPoints = new MatOfPoint();
        hull = new MatOfInt();
        hierarchy  = new Mat();
        touchedMat = new Mat();
        convexityDefects = new MatOfInt4();
        nonZero = new Mat();
        frame2 = new Mat();
        nonZeroRow = new Mat();
        contours = new ArrayList<>();
        palmCenter = new Point(-1, -1);
    }
        /**
     * Method to set scale factors for coordinate translation
     */
    protected void setScaleFactors(int vidWidth, int vidHeight){
        float deviceWidth = javaCameraView.getWidth();
        float deviceHeight = javaCameraView.getHeight();
        if(deviceHeight - vidHeight < deviceWidth - vidWidth){
            float temp = vidWidth * deviceHeight / vidHeight;
            offsetFactY = 0;
            offsetFactX = (deviceWidth - temp) / 2;
            scaleFactY = vidHeight / deviceHeight;
            scaleFactX = vidWidth / temp;
        }
        else{
            float temp = vidHeight * deviceWidth / vidWidth;
            offsetFactX= 0;
            offsetFactY = (deviceHeight - temp) / 2;
            scaleFactX = vidWidth / deviceWidth;
            scaleFactY = vidHeight / temp;
        }
    }


    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        // called for every frame on video feed

        ycc = inputFrame.rgba();
        if (handDetected) {
            // clone frame because original frame needed for display
            frame = ycc.clone();

            // remove noise and convert to binary in HSV range determined by user input
            Imgproc.GaussianBlur(frame, frame, new Size(9, 9), 5);
            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGB2HSV_FULL);
            Core.inRange(frame, minHSV, maxHSV, frame);

//            Point palm = getDistanceTransformCenter(frame);

            // get all possible contours and then determine palm contour
            contours =  getAllContours(frame);
            int indexOfPalmContour = getPalmContour(contours);

            if(indexOfPalmContour < 0)
                Log.e("palm", "no palm in the frame");  // no palm in frame
            else{
                // get anchor point
                Point palm = getDistanceTransformCenter(frame);
                Rect roi = Imgproc.boundingRect(contours.get(indexOfPalmContour));
                Imgproc.rectangle(ycc, roi.tl(), roi.br(), new Scalar(255,0,0),1);

            }
            return ycc;
        }
        return ycc;
    }
    @Override
    public void onCameraViewStopped() {
        // release all resources on camera close
        frame.release();
        ycc.release();
        ranges.release();
        channels.release();
        dstBackProject.release();
        palmContour.release();
        hullPoints.release();
        hull.release();
        hierarchy.release();
        touchedMat.release();
        convexityDefects.release();
        nonZero.release();
        frame2.release();
        nonZeroRow.release();
        while (allRoiHist.size() > 0)
            allRoiHist.get(0).release();
        while (contours.size() > 0)
            contours.get(0).release();
    }
    @Override
    public boolean onTouchEvent(MotionEvent event) {
        if(! handDetected){

            // clone and blur touched frame
            frame = ycc.clone();
            Imgproc.GaussianBlur(frame, frame, new Size(9, 9), 5);

            // calculate x, y coords because resolution is scaled on device display
            int x = Math.round((event.getX() - offsetFactX) * scaleFactX) ;
            int y = Math.round((event.getY() - offsetFactY) * scaleFactY);

            int rows = frame.rows();
            int cols = frame.cols();

            // return if touched point is outside camera resolution
            if ((x < 0) || (y < 0) || (x > cols) || (y > rows)) return false;

            // set palm center point and average HSV value
            palmCenter.x = x;
            palmCenter.y = y;

            getAvgHSV(frame);

            handDetected = true;
        }
        return false;
    }

        /**
     * Method to get all possible contours in binary image frame.
     */
    protected List<MatOfPoint> getAllContours(Mat frame){
        frame2 = frame.clone();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(frame2, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        return contours;
    }

    /**
     * Method to get contour of palm. Computed by the
     * knowledge that palm center has to lie inside it.
     */
    protected int getPalmContour(List<MatOfPoint> contours){

        Rect roi;
        int indexOfMaxContour = -1;
        for (int i = 0; i < contours.size(); i++) {
            roi = Imgproc.boundingRect(contours.get(i));
            if(roi.contains(palmCenter))
                return i;
        }
        return indexOfMaxContour;
    }

    /**
     * Method to assign average HSV value of palm
     */
    protected void getAvgHSV(Mat frame){

        // consider square patch around touched pixel
        int x = (int) palmCenter.x;
        int y = (int) palmCenter.y;
        int rows = frame.rows();
        int cols = frame.cols();

        Rect touchedSquare = new Rect();
        int squareSide = 20;

        touchedSquare.x = (x > squareSide) ? x - squareSide : 0;
        touchedSquare.y = (y > squareSide) ? y - squareSide : 0;

        touchedSquare.width = (x + squareSide < cols) ?
                x + squareSide - touchedSquare.x : cols - touchedSquare.x;
        touchedSquare.height = (y + squareSide < rows) ?
                y + squareSide - touchedSquare.y : rows - touchedSquare.y;

        touchedMat = frame.submat(touchedSquare);

        // convert patch to HSV and get average values
        Imgproc.cvtColor(touchedMat, touchedMat, Imgproc.COLOR_RGB2HSV_FULL);

        Scalar sumHSV = Core.sumElems(touchedMat);
        int total = touchedSquare.width * touchedSquare.height;
        double avgHSV[] = {sumHSV.val[0] / total, sumHSV.val[1] / total, sumHSV.val[2] / total};
        assignHSV(avgHSV);
    }

    /**
     * Method to assign range of HSV values of palm
     */
    protected void assignHSV(double avgHSV[]){
        minHSV.val[0] = (avgHSV[0] > 10) ? avgHSV[0] - 10 : 0;
        maxHSV.val[0] = (avgHSV[0] < 245) ? avgHSV[0] + 10 : 255;

        minHSV.val[1] = (avgHSV[1] > 130) ? avgHSV[1] - 100 : 30;
        maxHSV.val[1] = (avgHSV[1] < 155) ? avgHSV[1] + 100 : 255;

        minHSV.val[2] = (avgHSV[2] > 130) ? avgHSV[2] - 100 : 30;
        maxHSV.val[2] = (avgHSV[2] < 155) ? avgHSV[2] + 100 : 255;

        Log.e("HSV", avgHSV[0]+", "+avgHSV[1]+", "+avgHSV[2]);
        Log.e("HSV", minHSV.val[0]+", "+minHSV.val[1]+", "+minHSV.val[2]);
        Log.e("HSV", maxHSV.val[0]+", "+maxHSV.val[1]+", "+maxHSV.val[2]);
    }

    /**
     * Method to compute and return strongest point of distance transform.
     * For a binary image with palm in white, strongest point will be the palm center.
     */
    protected Point getDistanceTransformCenter(Mat frame){

        Imgproc.distanceTransform(frame, frame, Imgproc.CV_DIST_L2, 3);
        frame.convertTo(frame, CvType.CV_8UC1);
        Core.normalize(frame, frame, 0, 255, Core.NORM_MINMAX);
        Imgproc.threshold(frame, frame, 254, 255, Imgproc.THRESH_TOZERO);
        Core.findNonZero(frame, nonZero);

        // have to manually loop through matrix to calculate sums
        int sumx = 0, sumy = 0;
        for(int i=0; i<nonZero.rows(); i++) {
            sumx += nonZero.get(i, 0)[0];
            sumy += nonZero.get(i, 0)[1];
        }
        sumx /= nonZero.rows();
        sumy /= nonZero.rows();

        return new Point(sumx, sumy);
    }


}

























































