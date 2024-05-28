using AITest.Models;

namespace AITest;

internal class AITest
{
    private static void Main()
    {
        var network = new NeuralNetwork();
        var isOpen = true;
        Console.CursorVisible = false;


        while (isOpen)
        {
            Console.WriteLine("1. Обучение нейросети" +
                              "\n2. Тест нейросети по тренировочному набору" +
                              "\n3. Тест нейросети по входным данным" +
                              "\n4. Выход");

            var input = Console.ReadKey(true);

            switch (input.KeyChar)
            {
                case '1':
                    Console.Clear();
                    Console.WriteLine("\nИдёт процесс обучения...");
                    WriteError(network.Train());
                    break;
                case '2':
                    Console.Clear();
                    Console.WriteLine("\nИдёт процесс получения результатов...");
                    WriteResults(network.Test());
                    break;
                case '3':
                    Console.Clear();
                    var statements = new decimal[2];
                    if (statements == null) throw new ArgumentNullException(nameof(statements));
                    
                    Console.Write("Введите истинность первого выражения (0 или 1): ");
                    statements[0] = Convert.ToDecimal(Console.ReadLine());
                    
                    Console.Write("Введите истинность второго выражения (0 или 1): ");
                    statements[1] = Convert.ToDecimal(Console.ReadLine());
                    
                    Console.WriteLine("\nИдёт процесс получения результатов...");
                    WriteResults(network.TestManually(statements));
                    break;
                case '4':
                    Console.Clear();
                    isOpen = false;
                    break;
                default:
                    Console.Clear();
                    Console.WriteLine("Неверный ввод. Повторите попытку");
                    break;
            }

            if (input.KeyChar == '4') continue;
            Console.WriteLine("Нажмите любую клавишу для продолжения...\n");
            Console.ReadKey();
            Console.Clear();
        }
    }

    private static void WriteResults(decimal[,] results)
    {
        Console.WriteLine();
        Console.WriteLine("Результаты: \n");
        for (var i = 0; i < results.GetUpperBound(0) + 1; i++)
        {
            for (var j = 0; j < results.GetUpperBound(1) + 1; j++)
                Console.WriteLine($"{results[i, j]} \t");
            Console.WriteLine();
        }

        Console.WriteLine();
    }

    private static void WriteError(List<decimal> errors)
    {
        Console.WriteLine();
        Console.WriteLine("Процесс обучения окончен \n");
        Console.WriteLine("Журнал обучения: \n");
        foreach (var error in errors)
        {
            Console.WriteLine($"Ошибка эпохи: {error}");
        }

        Console.WriteLine("\nКонец обучения");
        Console.WriteLine();
    }
}