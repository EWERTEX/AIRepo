using System.Globalization;
using System.Xml;
using AITest.Models.Enums;

namespace AITest.Models.Layers;

public abstract class Layer
{
    protected Layer(int numberOfNeurons, int numberOfPreviousNeurons, NeuronType type, string typeString)
    {
        NumberOfNeurons = numberOfNeurons;
        NumberOfPreviousNeurons = numberOfPreviousNeurons;
        Neurons = new Neuron[NumberOfNeurons];
        var weights = WeightInitialize(MemoryMode.Get, typeString);

        for (var i = 0; i < numberOfNeurons; ++i)
        {
            var temporaryWeights = new decimal[numberOfPreviousNeurons];
            for (var j = 0; j < numberOfPreviousNeurons; ++j)
                temporaryWeights[j] = weights[i, j];
            Neurons[i] = new Neuron(null, temporaryWeights, type);
        }
    }

    protected readonly int NumberOfNeurons;
    protected readonly int NumberOfPreviousNeurons;
    protected const decimal LearningRate = (decimal)0.01;

    protected Neuron[] Neurons { get; set; }

    public decimal[] Data
    {
        set
        {
            foreach (var neuron in Neurons)
                neuron.Inputs = value;
        }
    }

    public decimal[,] WeightInitialize(MemoryMode memoryMode, string type)
    {
        var weights = new decimal[NumberOfNeurons, NumberOfPreviousNeurons];
        var memoryDocument = new XmlDocument();
        memoryDocument.Load($"{type}Memory.xml");
        var memoryElement = memoryDocument.DocumentElement;

        switch (memoryMode)
        {
            case MemoryMode.Get:
                for (var l = 0; l < weights.GetLength(0); ++l)
                for (var k = 0; k < weights.GetLength(1); ++k)
                    weights[l, k] = decimal
                        .Parse(
                            memoryElement?.ChildNodes.Item(k + weights.GetLength(1) * l)?.InnerText.Replace(',', '.') ??
                            string.Empty,
                            CultureInfo.InvariantCulture);
                break;

            case MemoryMode.Set:
                for (var l = 0; l < Neurons.Length; ++l)
                for (var k = 0; k < NumberOfPreviousNeurons; ++k)
                    memoryElement.ChildNodes.Item(k + NumberOfPreviousNeurons * l).InnerText =
                        Neurons[l].Weights[k].ToString(CultureInfo.InvariantCulture);
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(memoryMode), memoryMode, null);
        }

        memoryDocument.Save($"{type}Memory.xml");
        return weights;
    }

    public abstract void Recognize(NeuralNetwork? network, Layer? nextLayer);
    public abstract decimal[]? BackwardPass(decimal[] gradientsSums);
}