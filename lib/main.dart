import 'dart:math';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/linalg.dart';

import 'detector_view.dart';
import 'pose_painter.dart';
void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a blue toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: PoseDetectorView(),
    );
  }
}


class PoseDetectorView extends StatefulWidget {
  @override
  State<StatefulWidget> createState() => _PoseDetectorViewState();
}

class _PoseDetectorViewState extends State<PoseDetectorView> {
   List<List<double>> featuresList1=[];
   List<List<double>> finalList=[];
   List<List<double>> featuresList2=[];
  final PoseDetector _poseDetector =
  PoseDetector(options: PoseDetectorOptions());
  bool lebel1Activated=true;
  bool _canProcess = true;
  bool _isBusy = false;
  CustomPaint? _customPaint;
  String? _text;
  var _cameraLensDirection = CameraLensDirection.front;
  late KnnClassifier knn;

  bool testing=false;

  String _prediction='Lebel 1';

  @override
  void dispose() async {
    _canProcess = false;
    _poseDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(onPressed: (){
                  lebel1Activated=true;
                  testing=false;
                  startTraining();
                  setState(() {
                    
                  });
                }, child: Text('Lebel 1')),
                ElevatedButton(onPressed: (){
                  lebel1Activated=false;
                  testing=false;
                  startTraining();
                }, child: Text('Lebel 2')),
              ],
            ),
            ElevatedButton(onPressed: (){

              var listLength = min(featuresList1.length,featuresList2.length);
              // print(featuresList1[10].length.toString());
              var d1=Matrix.fromList(featuresList1,dtype: DType.float64).asFlattenedList;
              var d2=Matrix.fromList(featuresList2,dtype: DType.float64).asFlattenedList;
              // print('D1 = ${d1.length.toString()}');
              // print('D2 = ${d2.length.toString()}');
              // final data = [d1,d2];
              // int i,j,rows=2,cols=66;
              // List<List<dynamic>> newData= List<List>.generate(66, (i) => List<dynamic>.generate(2, (index) => null, growable: false), growable: false);
              // print(newData.toString());
              // for(i=0; i<rows; i++){
              //   for(j=0; j<cols; j++){
              //     newData[j][i] = data[i][j];
              //   }
              // }
              print(finalList.toString());
              DataFrame df=DataFrame(finalList,headerExists: false);
              ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Trained')));
              knn = KnnClassifier(df, 'col_66',6,);
            }, child: Text('Train')),
            ElevatedButton(onPressed: (){
              testing=true;
              startTraining();
            }, child: Text('Test'))
          ],
        ),
      ),
    );
  }

  Future<void> _processImage(InputImage inputImage) async {
    if (!_canProcess) return;
    if (_isBusy) return;
    _isBusy = true;
    setState(() {
      _text = '';
    });
    final poses = await _poseDetector.processImage(inputImage);
    List<List<double>> data=[];
    for(var pose in poses){
    data.add(extractLandmarks(pose));
    }
    if(lebel1Activated) {
      if(Matrix.fromList(data,dtype: DType.float64).asFlattenedList.isNotEmpty) {
        // featuresList1.add(Matrix.fromList(data,dtype: DType.float64).asFlattenedList);
        List<double> l1=Matrix.fromList(data,dtype: DType.float64).asFlattenedList.toList();
        l1.add(0);
        print(l1.last);
        finalList.add(l1);
      }
    } else {
      if(Matrix.fromList(data,dtype: DType.float64).asFlattenedList.isNotEmpty) {
        // featuresList2.add(Matrix.fromList(data,dtype: DType.float64).asFlattenedList);
        List<double> l1=Matrix.fromList(data,dtype: DType.float64).asFlattenedList.toList();
        l1.add(1);
        finalList.add(l1);
      }
    }
    if (inputImage.metadata?.size != null &&
        inputImage.metadata?.rotation != null) {
      final painter = PosePainter(
        poses,
        inputImage.metadata!.size,
        inputImage.metadata!.rotation,
        _cameraLensDirection,
      );
      _customPaint = CustomPaint(painter: painter);
    } else {
      _text = 'Poses found: ${poses.length}\n\n';
      // TODO: set _customPaint to draw landmarks on top of image
      _customPaint = null;
    }
    _isBusy = false;
    if (mounted) {
      setState(() {});
    }
  }
  Future<void> _testImage(InputImage inputImage) async {
    // if (!_canProcess) return;
    // if (_isBusy) return;
    // _isBusy = true;
    // setState(() {
    //   _text = '';
    // });
    print('Here in test');
    final poses = await _poseDetector.processImage(inputImage);
    List<List<double>> data=[];
    for(var pose in poses){
    data.add(extractLandmarks(pose));
    }
    var t1=Matrix.fromList(data).asFlattenedList.toList();
    print(t1);
    if(t1.isNotEmpty) {
      var df = DataFrame([t1],headerExists: false);
      var prediction = knn.predict(df);
      ScaffoldMessenger.of(context)
          .showSnackBar(SnackBar(content: Text('${prediction.toString()}')));
      print('Prediction done');
    }
    if (inputImage.metadata?.size != null &&
        inputImage.metadata?.rotation != null) {
      final painter = PosePainter(
        poses,
        inputImage.metadata!.size,
        inputImage.metadata!.rotation,
        _cameraLensDirection,
      );
      _customPaint = CustomPaint(painter: painter);
    } else {
      _text = 'Poses found: ${poses.length}\n\n';
      // TODO: set _customPaint to draw landmarks on top of image
      _customPaint = null;
    }
    _isBusy = false;
    if (mounted) {
      setState(() {});
    }
  }
  List<double> extractLandmarks(Pose pose) {
    List<double> landmarks = [];

    for (var landmarkType in PoseLandmarkType.values) {
      final landmark = pose.landmarks[landmarkType];
      landmarks.add(landmark!.x);
      landmarks.add(landmark.y);
      // Include z coordinate if you are using 3D pose detection
      // landmarks.add(landmark.position.z);
    }

    return landmarks;
  }

  void startTraining() async{
    await showDialog(context: context, builder: (builder){
     var number=3;
      return StatefulBuilder(builder: (BuildContext context, void Function(void Function()) setState) {
        Future.delayed(const Duration(seconds: 1)).then((value){
          number--;
          setState((){});
        });
        if(number==0) Navigator.pop(builder);
        return Scaffold(
          body: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text("$number",style: const TextStyle(fontSize: 80,fontWeight: FontWeight.w500),),
                ],
              )
            ],
          ),
        );
      },);
    });
    showDialog(context: context, builder: (builder){
     if(!testing) Future.delayed(Duration(seconds: 10), () {
        Navigator.of(context).pop(true);
      });
      return Scaffold(
        appBar: AppBar(),
        body: StatefulBuilder(
          builder: (BuildContext context, void Function(void Function()) setStat) {
            return Stack(
              alignment: AlignmentDirectional.bottomCenter,
              children: [
                DetectorView(
                  title: 'Pose Detector',
                  customPaint: _customPaint,
                  text: _text,
                  onImage: testing?(InputImage inputImage) async {
                    // if (!_canProcess) return;
                    // if (_isBusy) return;
                    // _isBusy = true;
                    // setState(() {
                    //   _text = '';
                    // });
                    print('Here in test');
                    final poses = await _poseDetector.processImage(inputImage);
                    List<List<double>> data=[];
                    for(var pose in poses){
                      data.add(extractLandmarks(pose));
                    }
                    var t1=Matrix.fromList(data).asFlattenedList.toList();
                    print(t1);
                    if(t1.isNotEmpty) {
                      var df = DataFrame([t1],headerExists: false);
                      var prediction = knn.predict(df);
                      print(prediction.toMatrix().asFlattenedList.last==1.0?"Two":'One');
                      _prediction=prediction.toMatrix().asFlattenedList.last==1.0?"Lebel 2":'Lebel 1';
                      print('Prediction done');
                    }
                    if (inputImage.metadata?.size != null &&
                        inputImage.metadata?.rotation != null) {
                      final painter = PosePainter(
                        poses,
                        inputImage.metadata!.size,
                        inputImage.metadata!.rotation,
                        _cameraLensDirection,
                      );
                      _customPaint = CustomPaint(painter: painter);
                    } else {
                      _text = 'Poses found: ${poses.length}\n\n';
                      // TODO: set _customPaint to draw landmarks on top of image
                      _customPaint = null;
                    }
                    _isBusy = false;
                    if (mounted) {
                      setStat(() {});
                    }
                  }: (InputImage inputImage) async {
                    if (!_canProcess) return;
                    if (_isBusy) return;
                    _isBusy = true;
                    setState(() {
                      _text = '';
                    });
                    final poses = await _poseDetector.processImage(inputImage);
                    List<List<double>> data=[];
                    for(var pose in poses){
                      data.add(extractLandmarks(pose));
                    }
                    if(lebel1Activated) {
                      if(Matrix.fromList(data,dtype: DType.float64).asFlattenedList.isNotEmpty) {
                        // featuresList1.add(Matrix.fromList(data,dtype: DType.float64).asFlattenedList);
                        List<double> l1=Matrix.fromList(data,dtype: DType.float64).asFlattenedList.toList();
                        l1.add(0);
                        print(l1.last);
                        finalList.add(l1);
                      }
                    } else {
                      if(Matrix.fromList(data,dtype: DType.float64).asFlattenedList.isNotEmpty) {
                        // featuresList2.add(Matrix.fromList(data,dtype: DType.float64).asFlattenedList);
                        List<double> l1=Matrix.fromList(data,dtype: DType.float64).asFlattenedList.toList();
                        l1.add(1);
                        finalList.add(l1);
                      }
                    }
                    if (inputImage.metadata?.size != null &&
                        inputImage.metadata?.rotation != null) {
                      final painter = PosePainter(
                        poses,
                        inputImage.metadata!.size,
                        inputImage.metadata!.rotation,
                        _cameraLensDirection,
                      );
                      _customPaint = CustomPaint(painter: painter);
                    } else {
                      _text = 'Poses found: ${poses.length}\n\n';
                      // TODO: set _customPaint to draw landmarks on top of image
                      _customPaint = null;
                    }
                    _isBusy = false;
                    if (mounted) {
                      setStat(() {});
                    }
                  },
                  initialCameraLensDirection: _cameraLensDirection,
                  onCameraLensDirectionChanged: (value) => _cameraLensDirection = value,
                ),
               if(testing) Container(
                   color: Colors.white,
                   margin: EdgeInsets.only(bottom: 20),
                   child: Text('Predicted Response : $_prediction',style: TextStyle(fontWeight: FontWeight.w500,fontSize: 18),))
              ],
            );
          },
        ),
      );});
    
  }
}
