using System.Globalization;
using System.Xml;
using System.Xml.Linq;
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
    protected const decimal LearningRate = (decimal)1;
    protected Neuron[] Neurons { get; set; }

    private int NumberOfLinks => NumberOfNeurons * NumberOfPreviousNeurons;

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
        var memoryName = $"{type}Memory.xml";
        
        var weights = new decimal[NumberOfNeurons, NumberOfPreviousNeurons];
        var memoryDocument = new XmlDocument();

        try
        {
            memoryDocument.Load(memoryName);
        }
        catch
        {
            var document = new XDocument();
            var documentWeights = new XElement("weights");
            
            for(var i = 0; i < NumberOfLinks; i++)
                documentWeights.Add(new XElement("weight", 0));
            
            document.Add(documentWeights);
            document.Save(memoryName);
            memoryDocument.Load(memoryName);
        }
        
        var memoryElement = memoryDocument.DocumentElement;

        if (memoryElement.ChildNodes.Count > NumberOfLinks)
        {
            memoryElement.RemoveAll();
            memoryDocument.Save(memoryName);
        }

        while (memoryElement.ChildNodes.Count < NumberOfLinks)
        {
            var weightElement = memoryDocument.CreateElement("weight");
            var weightText = memoryDocument.CreateTextNode("0");
            
            weightElement.AppendChild(weightText);
            memoryElement.AppendChild(weightElement);
            memoryDocument.Save(memoryName);
        }

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

    public abstract void Recognize(Layer? nextLayer = null, NeuralNetwork? network = null);
    public abstract decimal[]? BackwardPass(decimal[] gradientsSums);
}