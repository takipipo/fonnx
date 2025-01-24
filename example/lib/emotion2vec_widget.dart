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
      const emotion2vecModelPath = 'assets/models/emotion2vec/emotion2vec.onnx';
      const classifierModelPath = 'assets/models/classifier/409241894704317319_c4254ad62fdf43d1a541acbef4e1a0c2.onnx';
      
      // Sample audio data - replace with real audio input
      final sampleAudio = List.generate(16000, (index) => index % 2 == 0 ? 1 : -1);
      
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