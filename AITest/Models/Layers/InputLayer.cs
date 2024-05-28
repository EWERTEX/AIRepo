namespace AITest.Models.Layers;

public class InputLayer
{
    public (double[], double[])[] TrainingSet { get; } =
    [
        ([0, 0], [0, 1]),
        ([0, 1], [1, 0]),
        ([1, 0], [1, 0]),
        ([1, 1], [0, 1])
    ];
}