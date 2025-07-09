using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FirstAIModel
{
    public class SimpleModel
    {
        private Neuron[] Hidden;
        private Neuron OutputNeuron;

        public SimpleModel(int inputSize, int hiddenSize)
        {
            Hidden = new Neuron[hiddenSize];
            for (int i = 0; i < hiddenSize; i++)
                Hidden[i] = new Neuron(inputSize);

            OutputNeuron = new Neuron(hiddenSize);
        }

        public double Predict(double[] inputs)
        {
            double[] hiddenOutputs = new double[Hidden.Length];
            for (int i = 0; i < Hidden.Length; i++)
                hiddenOutputs[i] = Hidden[i].Activate(inputs);

            return OutputNeuron.Activate(hiddenOutputs);
        }

        public void Train(double[] inputs, double expected, double learningRate)
        {
            double prediction = Predict(inputs);
            double outputError = expected - prediction;

            OutputNeuron.Train(outputError, learningRate);

            for (int i = 0; i < Hidden.Length; i++)
            {
                double hiddenError = OutputNeuron.Weights[i] * outputError;
                Hidden[i].Train(hiddenError, learningRate);
            }
        }
        public void SaveModel(string path)
        {
            using (var writer = new StreamWriter(path))
            {
                foreach (var neuron in Hidden)
                    writer.WriteLine(neuron.ExportParameters());

                writer.WriteLine(OutputNeuron.ExportParameters());
            }
        }
        public void LoadModel(string path)
        {
            var lines = File.ReadAllLines(path);

            for (int i = 0; i < Hidden.Length; i++)
                Hidden[i].ImportParameters(lines[i]);

            OutputNeuron.ImportParameters(lines[Hidden.Length]);
        }


    }

}
