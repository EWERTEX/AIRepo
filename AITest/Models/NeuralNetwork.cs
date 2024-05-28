using AITest.Models.Enums;
using AITest.Models.Layers;
using static System.Math;

namespace AITest.Models;

public class NeuralNetwork
{
    private readonly InputLayer _inputLayer = new();
    private HiddenLayer _hiddenLayer = new(NumberOfHiddenNeurons, NumberOfInputNeurons, NeuronType.Hidden, nameof(_hiddenLayer));
    private OutputLayer _outputLayer = new(NumberOfOutputNeurons, NumberOfHiddenNeurons, NeuronType.Output, nameof(_outputLayer));
    private const int NumberOfInputNeurons = 2;
    private const int NumberOfHiddenNeurons = 6;
    private const int NumberOfOutputNeurons = 1;
    public readonly decimal[] Output = new decimal[NumberOfOutputNeurons];

    private void PassDirect()
    {
        _hiddenLayer.Recognize(_outputLayer);
        _outputLayer.Recognize(null,this);
    }
    
    private decimal GetMSE(IEnumerable<decimal> errors)
    {
        var sum = (decimal)errors.Sum(t => Pow((double)t, 2));

        return (decimal)0.5 * sum;
    }

    private decimal GetCost(IReadOnlyCollection<decimal> MSEs)
    {
        var sum = MSEs.Sum();

        return sum / MSEs.Count;
    }

    public List<decimal> Train()
    {
        List<decimal> outErrors = [];
        const decimal threshold = (decimal)0.001;
        var temporaryMSEs = new decimal[_inputLayer.TrainingSet.Length];
        decimal temporaryCost;

        do
        {
            for (var i = 0; i < _inputLayer.TrainingSet.Length; ++i)
            {
                _hiddenLayer.Data = _inputLayer.TrainingSet[i].Item1;
                PassDirect();

                var errors = new decimal[NumberOfOutputNeurons];
                for (var x = 0; x < errors.Length; ++x)
                    errors[x] = _inputLayer.TrainingSet[i].Item2[x] - Output[x];
                temporaryMSEs[i] = GetMSE(errors);
                var tempGradSum = _outputLayer.BackwardPass(errors);
                _hiddenLayer.BackwardPass(tempGradSum);
            }

            temporaryCost = GetCost(temporaryMSEs);
            //Console.WriteLine(temporaryCost); //для проверки сходимости в консоли убрать "//". По завершению проверок вернуть "//"
            outErrors.Add(temporaryCost);
        } while (temporaryCost > threshold);

        _hiddenLayer.WeightInitialize(MemoryMode.Set, nameof(_hiddenLayer));
        _outputLayer.WeightInitialize(MemoryMode.Set, nameof(_outputLayer));

        return outErrors;
    }

    public decimal[,] Test()
    {
        var result = new decimal[_inputLayer.TrainingSet.Length, NumberOfOutputNeurons];

        for (var i = 0; i < _inputLayer.TrainingSet.Length; ++i)
        {
            _hiddenLayer.Data = _inputLayer.TrainingSet[i].Item1;
            PassDirect();
            for (var j = 0; j < Output.Length; ++j)
                result[i, j] = Output[j];
        }

        return result;
    }

    public decimal[,] TestManually(decimal[] input)
    {
        var result = new decimal[1, NumberOfOutputNeurons];
        _hiddenLayer.Data = input;
        PassDirect();
        for (var j = 0; j < Output.Length; ++j)
            result[0, j] = Output[j];

        return result;
    }
}