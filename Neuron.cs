using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace FirstAIModel
{
    public class Neuron
    {
        public double[] Weights;
        public double Bias;
        public double Output;
        public double[] Inputs;

        private static Random rand = new Random();

        public Neuron(int inputCount)
        {
            Weights = new double[inputCount];
            for (int i = 0; i < inputCount; i++)
                Weights[i] = rand.NextDouble() * 2 - 1; // Random weights between -1 and 1
            Bias = rand.NextDouble() * 2 - 1;
        }

        public double Activate(double[] inputs)
        {
            Inputs = inputs;
            double sum = Bias;
            for (int i = 0; i < inputs.Length; i++)
                sum += inputs[i] * Weights[i];

            Output = Sigmoid(sum);
            return Output;
        }

        public void Train(double error, double learningRate)
        {
            double delta = error * SigmoidDerivative(Output);
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] += Inputs[i] * delta * learningRate;

            Bias += delta * learningRate;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));
        private double SigmoidDerivative(double y) => y * (1 - y);

        public string ExportParameters()
        {
            return string.Join(",", Weights.Select(w => w.ToString("F6"))) + "|" + Bias.ToString("F6");
        }

        public void ImportParameters(string data)
        {
            var parts = data.Split('|');
            var weightParts = parts[0].Split(',');
            for (int i = 0; i < weightParts.Length; i++)
                Weights[i] = double.Parse(weightParts[i]);

            Bias = double.Parse(parts[1]);
        }

    }

}
