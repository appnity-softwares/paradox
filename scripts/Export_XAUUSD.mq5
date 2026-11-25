//+------------------------------------------------------------------+
//|                                                Export_XAUUSD.mq5 |
//|                             AI Quant Terminal - Real Data Export |
//+------------------------------------------------------------------+
#property copyright "AI Quant Terminal"
#property link      "http://127.0.0.1:8123"
#property version   "1.00"
#property script_show_inputs

input string InpSymbol = "XAUUSD";
input ENUM_TIMEFRAMES InpTimeframe = PERIOD_M15;
input int InpBars = 10000;

void OnStart()
{
   Print("Exporting ", InpSymbol, " data for AI Quant Terminal...");
   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(InpSymbol, InpTimeframe, 0, InpBars, rates);
   if(copied <= 0) {
      Print("Failed to copy rates. Error: ", GetLastError());
      return;
   }
   string filename = InpSymbol + "_" + EnumToString(InpTimeframe) + ".csv";
   int handle = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
   if(handle == INVALID_HANDLE) {
      Print("Cannot open file: ", filename);
      return;
   }
   FileWrite(handle, "Time", "Open", "High", "Low", "Close", "Volume");
   for(int i = 0; i < copied; i++) {
      FileWrite(handle,
         TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES|TIME_SECONDS),
         DoubleToString(rates[i].open, 2),
         DoubleToString(rates[i].high, 2),
         DoubleToString(rates[i].low, 2),
         DoubleToString(rates[i].close, 2),
         IntegerToString(rates[i].tick_volume)
      );
   }
   FileClose(handle);
   Print("Successfully exported ", copied, " bars to MQL5/Files/", filename);
   Alert("AI Quant Terminal: Exported ", copied, " bars of ", InpSymbol);
}
