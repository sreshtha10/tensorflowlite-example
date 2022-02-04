package com.sreshtha.tensorflowlite_example

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.sreshtha.tensorflowlite_example.databinding.ActivityMainBinding
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {
    private lateinit var interpreter:Interpreter
    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        interpreter = Interpreter(loadModel(),null)
        binding.btnPredict.setOnClickListener {
            val str = binding.etInput.text.toString().trim()
            if(str.isNotEmpty()){
                binding.tvResult.text = doInference(str).toString()
            }
        }
    }



    private fun loadModel():MappedByteBuffer{
        val assetFileDescriptor = this.assets.openFd("linear.tflite")
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffSet = assetFileDescriptor.startOffset
        val length = assetFileDescriptor.length

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffSet,length)
    }

    private fun doInference(str:String):Float{
        val input = FloatArray(1)
        input[0] = str.toFloat()
        val output = Array(1){FloatArray(1)}
        interpreter.run(input,output)
        return output[0][0]
    }
}