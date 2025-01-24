import 'package:flutter/material.dart';
import 'package:fonnx/fonnx.dart';
import 'dart:typed_data';

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
      // Replace with your actual model path
      const emotion2vecModelPath = 'assets/emotion2vec/model.onnx';
      const classifierModelPath = 'assets/emotion2vec/classifier.onnx';
      
      // Sample audio data - replace with real audio input
      final sampleAudio = List.generate(16000, (index) => index % 2 == 0 ? 1 : -1);
      
      final result = await Fonnx().emotion2vec(
        emotion2vecModelPath: emotion2vecModelPath,
        classifierModelPath: classifierModelPath,
        audioBytes: sampleAudio,
      );

      setState(() {
        _result = result != null 
            ? 'Detected emotion: ${_interpretEmotion(result)}'
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

  String _interpretEmotion(Float32List emotions) {
    // Add your emotion labels here
    const emotions = ['Happy', 'Sad', 'Angry', 'Neutral'];
    
    // Find the highest probability emotion
    var maxIndex = 0;
    var maxValue = emotions[0];
    
    for (var i = 1; i < emotions.length; i++) {
      if (emotions[i] > maxValue) {
        maxValue = emotions[i];
        maxIndex = i;
      }
    }
    
    return emotions[maxIndex];
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