import 'package:flutter/material.dart';
import 'package:fonnx/fonnx.dart';
import 'dart:typed_data';
import 'package:path_provider/path_provider.dart'as path_provider; 
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'dart:io';
import 'package:path/path.dart' as path;

class Emotion2VecWidget extends StatefulWidget {
  const Emotion2VecWidget({super.key});

  @override
  State<Emotion2VecWidget> createState() => _Emotion2VecWidgetState();
}

class _Emotion2VecWidgetState extends State<Emotion2VecWidget> {
  String _result = 'No emotion detected yet';
  bool _isProcessing = false;

  Future<void> _processAudio() async {
    if (_isProcessing) return;

    setState(() {
      _isProcessing = true;
      _result = 'Processing audio...';
    });

    try {
      final emotion2vecModelPath = await getEmotion2VecModelPath('emotion2vec.onnx');
      final classifierModelPath = await getEmotion2VecModelPath('classifier.onnx');
      
      final sampleAudio = List.generate(16000 * 5, (index) => index % 2 == 0 ? 1 : -1);
      
      final result = await Fonnx().emotion2vec(
        emotion2vecModelPath: emotion2vecModelPath,
        classifierModelPath: classifierModelPath,
        audioBytes: sampleAudio,
      );

      // Print raw emotion vector to console
      print('Emotion vector: $result');

      setState(() {
        _result = result != null 
            ? 'Raw emotion vector: $result'
            : 'Failed to detect emotion';
      });
    } catch (e) {
      setState(() {
        _result = 'Error: $e';
      });
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  Future<String> getEmotion2VecModelPath(String modelFilenameWithExtension) async {
    if (kIsWeb) {
      return 'assets/models/emotion2vec/$modelFilenameWithExtension';
    }
    final assetCacheDirectory = await path_provider.getApplicationSupportDirectory();
    final modelPath = path.join(assetCacheDirectory.path, modelFilenameWithExtension);

    File file = File(modelPath);
    bool fileExists = await file.exists();
    final fileLength = fileExists ? await file.length() : 0;

    final assetPath = 'assets/models/emotion2vec/${path.basename(modelFilenameWithExtension)}';
    final assetByteData = await rootBundle.load(assetPath);
    final assetLength = assetByteData.lengthInBytes;
    final fileSameSize = fileLength == assetLength;

    if (!fileExists || !fileSameSize) {
      debugPrint('Copying emotion2vec model to $modelPath');
      List<int> bytes = assetByteData.buffer.asUint8List(
        assetByteData.offsetInBytes,
        assetByteData.lengthInBytes,
      );
      try {
        if (!fileExists) {
          await file.create(recursive: true);
        }
        await file.writeAsBytes(bytes, flush: true);
      } catch (e) {
        debugPrint('Error writing bytes to $modelPath: $e');
        rethrow;
      }
    }
    return modelPath;
  }

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Emotion2Vec Demo',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _isProcessing ? null : _processAudio,
              child: Text(_isProcessing ? 'Processing...' : 'Detect Emotion'),
            ),
            const SizedBox(height: 16),
            Text(_result),
          ],
        ),
      ),
    );
  }
} 