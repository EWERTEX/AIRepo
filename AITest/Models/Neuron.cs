using AITest.Models.Enums;
using static System.Math;

namespace AITest.Models;

public class Neuron(double[]? inputs, double[] weights, NeuronType type)
{
    public double Output => Inputs != null ? Activate(Inputs, Weights) : double.NaN;

    public double[] Weights { get; set; } = weights;
    public double[]? Inputs { get; set; } = inputs;

    private static double Activate(IEnumerable<double> inputs, IReadOnlyList<double> weights)
    {
        var sum = inputs.Select((t, i) => t * weights[i]).Sum();

        return Pow(1 + Exp(-sum), -1);
    }

    public double CalculateGradient(double error, double dif, double gradientsSum) =>
        (type == NeuronType.Output) ? error * dif : gradientsSum * dif;

    public double CalculateDerivative(double outSignal) => outSignal * (1 - outSignal);
}