using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading.Tasks;

internal static class Transport
{
    private static async Task CopyAsync(Stream input, Stream output)
    {
        byte[] buffer = new byte[8192];
        int length;
        while ((length = await input.ReadAsync(buffer, 0, buffer.Length)) != 0)
        {
            await output.WriteAsync(buffer, 0, length);
            await output.FlushAsync();
        }
    }

    private static string Quote(string value)
    {
        StringBuilder result = new StringBuilder("\"");
        int slashes = 0;
        foreach (char character in value)
        {
            if (character == '\\')
            {
                slashes++;
                continue;
            }
            result.Append('\\', character == '"' ? slashes * 2 + 1 : slashes);
            result.Append(character);
            slashes = 0;
        }
        result.Append('\\', slashes * 2);
        result.Append('"');
        return result.ToString();
    }

    public static int Main(string[] args)
    {
        string directory = AppDomain.CurrentDomain.BaseDirectory;
        string command = Path.GetFileNameWithoutExtension(Environment.GetCommandLineArgs()[0]);
        if (command == "askpass")
        {
            if (args.Length != 1 || !args[0].Contains("user@111.62.35.107") ||
                !args[0].ToLowerInvariant().Contains("password"))
                return 1;
            string memory = Path.Combine(Directory.GetParent(directory.TrimEnd('\\')).FullName, ".agents", "memory.md");
            const string prefix = "- SSH password: ";
            foreach (string line in File.ReadLines(memory, Encoding.UTF8))
            {
                if (line.StartsWith(prefix, StringComparison.Ordinal))
                {
                    Console.WriteLine(line.Substring(prefix.Length));
                    return 0;
                }
            }
            return 1;
        }
        if (command != "ssh" && command != "scp")
            return 2;
        string executable = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "OpenSSH", command + ".exe");
        StringBuilder arguments = new StringBuilder("-F ");
        arguments.Append(Quote(Path.Combine(directory, "ssh_config")));
        foreach (string argument in args)
        {
            arguments.Append(' ');
            string forwarded = argument;
            if (command == "ssh" && argument.StartsWith(".mutagen/agents/", StringComparison.Ordinal))
            {
                const string remoteData = "/data1/xjr/genesis-world-demos/mutagen";
                forwarded = "env MUTAGEN_DATA_DIRECTORY=" + remoteData + " " + remoteData + argument.Substring(8);
            }
            arguments.Append(Quote(forwarded));
        }
        ProcessStartInfo info = new ProcessStartInfo(executable, arguments.ToString());
        info.UseShellExecute = false;
        info.CreateNoWindow = true;
        info.RedirectStandardInput = true;
        info.RedirectStandardOutput = true;
        info.RedirectStandardError = true;
        info.EnvironmentVariables["SSH_ASKPASS"] = Path.Combine(directory, "askpass.exe");
        info.EnvironmentVariables["SSH_ASKPASS_REQUIRE"] = "force";
        info.EnvironmentVariables["DISPLAY"] = "genesis-remote";
        using (Process process = Process.Start(info))
        {
            Task input = CopyAsync(Console.OpenStandardInput(), process.StandardInput.BaseStream);
            input.ContinueWith(task => process.StandardInput.Close(), TaskContinuationOptions.OnlyOnRanToCompletion);
            Task output = CopyAsync(process.StandardOutput.BaseStream, Console.OpenStandardOutput());
            Task error = CopyAsync(process.StandardError.BaseStream, Console.OpenStandardError());
            process.WaitForExit();
            Task.WaitAll(output, error);
            return process.ExitCode;
        }
    }
}
