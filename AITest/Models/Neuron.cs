using AITest.Models.Enums;
using static System.Math;

namespace AITest.Models;

public class Neuron(decimal[]? inputs, decimal[] weights, NeuronType type)
{
    public decimal Output => Inputs != null ? Activate(Inputs, Weights) : 0;

    public decimal[] Weights { get; set; } = weights;
    public decimal[]? Inputs { get; set; } = inputs;

    private static decimal Activate(IEnumerable<decimal> inputs, IReadOnlyList<decimal> weights)
    {
        var sum = inputs.Select((t, i) => t * weights[i]).Sum();

        return (decimal)Pow(1 + Exp((double)-sum), -1);
    }

    public decimal CalculateGradient(decimal error, decimal dif, decimal gradientsSum) =>
        (type == NeuronType.Output) ? error * dif : gradientsSum * dif;

    public decimal CalculateDerivative(decimal outSignal) => outSignal * (1 - outSignal);
}