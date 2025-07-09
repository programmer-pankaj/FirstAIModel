using FirstAIModel;

class Program
{
    static void Main(string[] args)
    {
        var model = new SimpleModel(2, 4); // 2 inputs, 4 hidden neurons
        var learningRate = 0.1;
        int max = 10;
        int maxSum = 2 * max;
        Random rand = new Random();
        // Train
        for (int epoch = 0; epoch < 10000; epoch++)
        {
            for (int i = 0; i < 100; i++)
            {
                double a = rand.NextDouble() * max;
                double b = rand.NextDouble() * max;

                double[] inputs = { a / maxSum, b / maxSum };
                double expected = (a + b) / maxSum;

                model.Train(inputs, expected, learningRate);
            }
        }
        model.SaveModel("model.txt"); // after training

        // Later...
       // model.LoadModel("model.txt"); // restore trained model

        // Test
        for (int i = 0; i < 5; i++)
        {
            double a = rand.NextDouble() * max;
            double b = rand.NextDouble() * max;
            double[] inputs = { a / maxSum, b / maxSum };

            double prediction = model.Predict(inputs) * maxSum;
            double actual = a + b;

            Console.WriteLine($"{a:F2} + {b:F2} = {prediction:F2} (actual: {actual:F2})");
        }

        Console.ReadLine();
    }
}
