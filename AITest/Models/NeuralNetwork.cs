using AITest.Models.Enums;
using AITest.Models.Layers;
using static System.Math;

namespace AITest.Models;

public class NeuralNetwork
{
    private readonly InputLayer _inputLayer = new();
    private HiddenLayer _hiddenLayer = new(4, 2, NeuronType.Hidden, nameof(_hiddenLayer));
    private OutputLayer _outputLayer = new(2, 4, NeuronType.Output, nameof(_outputLayer));
    public readonly double[] Output = new double[2];

    private void PassDirect()
    {
        _hiddenLayer.Recognize(null, _outputLayer);
        _outputLayer.Recognize(this, null);
    }
    
    private double GetMSE(IEnumerable<double> errors)
    {
        var sum = errors.Sum(t => Pow(t, 2));

        return 0.5d * sum;
    }

    private double GetCost(IReadOnlyCollection<double> MSEs)
    {
        var sum = MSEs.Sum();

        return sum / MSEs.Count;
    }

    public List<double> Train()
    {
        List<double> outErrors = [];
        const double threshold = 0.001d;
        var temporaryMSEs = new double[4];
        double temporaryCost;

        do
        {
            for (var i = 0; i < _inputLayer.TrainingSet.Length; ++i)
            {
                _hiddenLayer.Data = _inputLayer.TrainingSet[i].Item1;
                PassDirect();

                var errors = new double[_inputLayer.TrainingSet[i].Item2.Length];
                for (var x = 0; x < errors.Length; ++x)
                    errors[x] = _inputLayer.TrainingSet[i].Item2[x] - Output[x];
                temporaryMSEs[i] = GetMSE(errors);
                var tempGradSum = _outputLayer.BackwardPass(errors);
                _hiddenLayer.BackwardPass(tempGradSum);
            }

            temporaryCost = GetCost(temporaryMSEs);
            outErrors.Add(temporaryCost);
        } while (temporaryCost > threshold);

        _hiddenLayer.WeightInitialize(MemoryMode.Set, nameof(_hiddenLayer));
        _outputLayer.WeightInitialize(MemoryMode.Set, nameof(_outputLayer));

        return outErrors;
    }

    public double[,] Test()
    {
        var result = new double[4, 2];

        for (var i = 0; i < _inputLayer.TrainingSet.Length; ++i)
        {
            _hiddenLayer.Data = _inputLayer.TrainingSet[i].Item1;
            PassDirect();
            for (var j = 0; j < Output.Length; ++j)
                result[i, j] = Output[j];
        }

        return result;
    }

    public double[,] TestManually(double[] input)
    {
        var result = new double[1, 2];
        _hiddenLayer.Data = input;
        PassDirect();
        for (var j = 0; j < Output.Length; ++j)
            result[0, j] = Output[j];

        return result;
    }
}