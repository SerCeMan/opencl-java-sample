package ru.serce;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.nio.FloatBuffer;
import java.util.Scanner;

import static com.jogamp.opencl.CLMemory.Mem.READ_ONLY;
import static com.jogamp.opencl.CLMemory.Mem.READ_WRITE;

/**
 * ./gradlew shadow && java -jar build/libs/opencl2-1.0-SNAPSHOT-all.jar
 */
public class Main {

    private static final Scanner in;
    private static PrintStream out;

    static {
        FileInputStream result;
        FileOutputStream output;
        try {
            result = new FileInputStream("input.txt");
            output = new FileOutputStream("output.txt");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        in = new Scanner(result);
        out = new PrintStream(output);
    }

    public static void main(String[] args) throws IOException {

        // set up (uses default CLPlatform and creates context for all devices)
        CLContext context = CLContext.create();
        System.out.println("created " + context);

        // always make sure to release the context under all circumstances
        // not needed for this particular sample but recommented
        try {

            // select fastest device
            CLDevice device = context.getMaxFlopsDevice();
            System.out.println("using " + device);

            // create command queue on device.
            CLCommandQueue queue = device.createCommandQueue();

            InputStream input = Main.class.getResourceAsStream("/VectorAdd.cl");
            CLProgram program = context.createProgram(input).build();

            int blockSize = 256;
//            int NSIZE = 50000;
            int NSIZE = in.nextInt();
            float[] inputarr = new float[NSIZE];
            int arrSize = NSIZE;
            if (NSIZE % blockSize != 0) {
                arrSize = (NSIZE / blockSize) + 1;
                arrSize *= blockSize;
            }
            for (int i = 0; i < NSIZE; ++i) {
//                inputarr[i] = 1;//in.nextFloat();
                inputarr[i] = in.nextFloat();
            }

            float[] outputarr = new float[arrSize];
            int sumSize = arrSize / blockSize;
            if (sumSize > blockSize
                    && sumSize % blockSize != 0) {
                sumSize = (sumSize / 256 + 1);
                sumSize *= 256;
            }
            float[] summand_vector = new float[sumSize];

            CLBuffer<FloatBuffer> clInputBuf = context.createFloatBuffer(inputarr.length, READ_ONLY);
            CLBuffer<FloatBuffer> clOutputBuf = context.createFloatBuffer(outputarr.length, READ_WRITE);

            CLBuffer<FloatBuffer> clSumBuf = context.createFloatBuffer(summand_vector.length, READ_WRITE);
            CLBuffer<FloatBuffer> clSumBufA = context.createFloatBuffer(summand_vector.length, READ_WRITE);

            fillBuffer(clInputBuf, inputarr);
            fillBuffer(clSumBuf, summand_vector);

            queue.putWriteBuffer(clInputBuf, true)
                    .putWriteBuffer(clSumBuf, true);

            CLKernel scanKernel = program.createCLKernel("block_scan");
            CLKernel sumKernel = program.createCLKernel("sum_scan");
            CLKernel addKernel = program.createCLKernel("add_scan");

            scanKernel.putArgs(clInputBuf, clOutputBuf, clSumBuf)
                    .putNullArg(4 * blockSize)
                    .putNullArg(4 * blockSize);
            queue.put1DRangeKernel(scanKernel, 0, arrSize, blockSize);
            if (sumSize > blockSize) {
                int sz = sumSize / blockSize;
                float[] tmpArr = new float[sumSize];
                CLBuffer<FloatBuffer> clTBuf = context.createFloatBuffer(summand_vector.length, READ_ONLY);
                CLBuffer<FloatBuffer> clTSumBuf = context.createFloatBuffer(sz, READ_ONLY);
                CLBuffer<FloatBuffer> clTSumBufP = context.createFloatBuffer(sz, READ_ONLY);

                scanKernel.putArgs(clSumBuf, clTBuf, clTSumBuf)
                        .putNullArg(4 * blockSize)
                        .putNullArg(4 * blockSize);
                queue.put1DRangeKernel(scanKernel, 0, sz, sz);

                sumKernel
                        .putArgs(clTSumBuf, clTSumBufP)
                        .putNullArg(4 * sz)
                        .putNullArg(4 * sz);
                queue.put1DRangeKernel(sumKernel, 0, sumSize, blockSize);

                addKernel.putArgs(clTBuf, clTSumBufP);
                queue.put1DRangeKernel(addKernel, 0, sumSize, blockSize);

                fillBuffer(clTBuf, tmpArr);
                queue.putReadBuffer(clTBuf, true);

                fillBuffer(clSumBufA, tmpArr);
                queue.putWriteBuffer(clSumBufA, true);
            } else {
                sumKernel
                        .putArgs(clSumBuf, clSumBufA)
                        .putNullArg(4 * sumSize)
                        .putNullArg(4 * sumSize);
                queue.put1DRangeKernel(sumKernel, 0, sumSize, sumSize);
            }
            addKernel
                    .putArg(clOutputBuf)
                    .putArg(clSumBufA);
            queue.put1DRangeKernel(addKernel, 0, arrSize, blockSize);

            queue.putReadBuffer(clOutputBuf, true);
            clOutputBuf.getBuffer().get(outputarr);
            for (int i = 0; i < NSIZE; ++i) {
                out.print(outputarr[i] + " ");
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            // cleanup all resources associated with this context.
            context.release();
        }
    }

    private static void fillBuffer(CLBuffer<FloatBuffer> cbuffer, float[] arr) {
        FloatBuffer buffer = cbuffer.getBuffer();
        buffer.mark();
        for (float v : arr) {
            buffer.put(v);
        }
        buffer.reset();
    }
}