import 'package:flutter/services.dart';
import 'dart:typed_data';

class Emotion2Vec {
  static const MethodChannel _channel = MethodChannel('emotion2vec');
  
  Future<Map<String, dynamic>?> processAudio(List<double> audioData) async {
    try {
      // Convert double list to Float32List
      final Float32List audioFloat32 = Float32List.fromList(audioData);
      
      final result = await _channel.invokeMethod('processAudio', {
        'audioData': audioFloat32,
      });
      
      return result as Map<String, dynamic>?;
    } catch (e) {
      print('Error processing audio: $e');
      return null;
    }
  }
} 