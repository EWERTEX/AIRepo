using AITest.Models.Enums;

namespace AITest.Models.Layers;

public class OutputLayer(int numberOfNeurons, int numberOfPreviousNeurons, NeuronType type, string typeString)
    : Layer(numberOfNeurons, numberOfPreviousNeurons, type, typeString)
{
    public override void Recognize(NeuralNetwork? network, Layer? nextLayer)
    {
        for (var i = 0; i < Neurons.Length; ++i)
            if (network != null)
                network.Output[i] = Neurons[i].Output;
    }

    public override decimal[] BackwardPass(decimal[] errors)
    {
        var gradientSum = new decimal[NumberOfPreviousNeurons];

        for (var j = 0; j < gradientSum.Length; ++j)
        {
            var sum = Neurons.Select((t, k) => t.Weights[j] * t.CalculateGradient(errors[k], t.CalculateDerivative(t.Output), 0)).Sum();
            gradientSum[j] = sum;
        }
        
        for (var i = 0; i < NumberOfNeurons; ++i)
            for (var n = 0; n < NumberOfPreviousNeurons; ++n)
            {
                var inputs = Neurons[i].Inputs;
                if (inputs != null)
                    Neurons[i].Weights[n] += LearningRate * inputs[n] * Neurons[i]
                        .CalculateGradient(errors[i], Neurons[i].CalculateDerivative(Neurons[i].Output), 0);
            }

        return gradientSum;
    }
}