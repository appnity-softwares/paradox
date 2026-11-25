//+------------------------------------------------------------------+
//|                                      Export_XAUUSD_Ticks.mq5     |
//|          AI Quant Terminal - real bid/ask tick history export    |
//|                                                                  |
//|  Exports genuine broker ticks with CopyTicksRange, one CSV per   |
//|  month, into MQL5/Files/ticks/. The Python pipeline watches that |
//|  folder, converts each month to partitioned Parquet, then        |
//|  deletes the CSV so disk usage stays bounded.                    |
//|                                                                  |
//|  Nothing is synthesised: if the broker has no ticks for a range  |
//|  the month is reported as EMPTY, never filled with fake data.    |
//+------------------------------------------------------------------+
#property copyright "AI Quant Terminal"
#property version   "1.00"
#property script_show_inputs

input string   InpSymbol      = "XAUUSD";
input int      InpYearsBack   = 2;        // how many years of history to export
input bool     InpInfoTicks   = true;     // true = quote ticks (bid/ask), false = all ticks
input bool     InpSkipExisting= true;     // resume: skip months already exported
input int      InpPauseMs     = 20;       // pause between days (keeps the terminal responsive)
input int      InpMaxRetries  = 6;        // retries while the broker downloads history

string g_dir;
long   g_total_ticks = 0;
int    g_months_done = 0;
int    g_months_empty = 0;

void WriteStatus(const string state, const string detail)
{
   int h = FileOpen(g_dir + "\\_export_status.txt", FILE_WRITE|FILE_TXT|FILE_ANSI);
   if(h == INVALID_HANDLE) return;
   FileWrite(h, "state=" + state);
   FileWrite(h, "symbol=" + InpSymbol);
   FileWrite(h, "detail=" + detail);
   FileWrite(h, "total_ticks=" + IntegerToString(g_total_ticks));
   FileWrite(h, "months_done=" + IntegerToString(g_months_done));
   FileWrite(h, "months_empty=" + IntegerToString(g_months_empty));
   FileWrite(h, "updated=" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS));
   FileClose(h);
}

// Copy one day of ticks. Returns the number of ticks, or -1 on hard failure.
int CopyDay(MqlTick &ticks[], const datetime day_start)
{
   ulong from_msc = (ulong)day_start * 1000;
   ulong to_msc   = from_msc + 86400000 - 1;
   uint  flags    = InpInfoTicks ? COPY_TICKS_INFO : COPY_TICKS_ALL;

   for(int attempt = 0; attempt < InpMaxRetries; attempt++)
   {
      ResetLastError();
      int copied = CopyTicksRange(InpSymbol, ticks, flags, from_msc, to_msc);
      if(copied > 0) return copied;

      int err = GetLastError();
      // 4401/4402/4403: history is still being downloaded from the broker.
      if(err == 4401 || err == 4402 || err == 4403 || err == 0)
      {
         Sleep(1500);
         continue;
      }
      return -1;
   }
   return 0;   // broker has no ticks for this day
}

void OnStart()
{
   g_dir = "ticks";
   FolderCreate(g_dir, 0);

   datetime now  = TimeCurrent();
   datetime from = now - (datetime)(InpYearsBack * 365 * 86400);

   MqlDateTime dt;
   TimeToStruct(from, dt);
   dt.day = 1; dt.hour = 0; dt.min = 0; dt.sec = 0;
   datetime cursor = StructToTime(dt);

   Print("=== AI Quant Terminal: exporting ", InpYearsBack, " years of ", InpSymbol, " ticks ===");
   WriteStatus("STARTING", TimeToString(cursor, TIME_DATE));

   while(cursor < now)
   {
      MqlDateTime c;
      TimeToStruct(cursor, c);
      int year = c.year, month = c.mon;

      // first day of next month
      MqlDateTime n = c;
      n.mon = month + 1;
      if(n.mon > 12) { n.mon = 1; n.year = year + 1; }
      n.day = 1; n.hour = 0; n.min = 0; n.sec = 0;
      datetime next_month = StructToTime(n);
      datetime month_end  = (next_month < now) ? next_month : now;

      string fname = g_dir + "\\" + InpSymbol + "_ticks_"
                   + IntegerToString(year) + "_"
                   + (month < 10 ? "0" : "") + IntegerToString(month) + ".csv";

      if(InpSkipExisting && FileIsExist(fname))
      {
         Print("skip (already exported): ", fname);
         cursor = next_month;
         g_months_done++;
         continue;
      }

      int handle = FileOpen(fname, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
      if(handle == INVALID_HANDLE)
      {
         Print("cannot open ", fname, " err=", GetLastError());
         WriteStatus("ERROR", "cannot open " + fname);
         return;
      }
      FileWrite(handle, "time_msc", "bid", "ask", "last", "volume", "flags");

      long month_ticks = 0;
      datetime day = cursor;
      while(day < month_end)
      {
         MqlTick ticks[];
         int copied = CopyDay(ticks, day);
         if(copied > 0)
         {
            for(int i = 0; i < copied; i++)
            {
               FileWrite(handle,
                  IntegerToString(ticks[i].time_msc),
                  DoubleToString(ticks[i].bid, 5),
                  DoubleToString(ticks[i].ask, 5),
                  DoubleToString(ticks[i].last, 5),
                  IntegerToString((long)ticks[i].volume),
                  IntegerToString(ticks[i].flags));
            }
            month_ticks += copied;
         }
         day += 86400;
         Sleep(InpPauseMs);
      }
      FileClose(handle);

      g_total_ticks += month_ticks;
      if(month_ticks == 0)
      {
         g_months_empty++;
         FileDelete(fname);
         Print("EMPTY month (broker has no ticks): ", year, "-", month);
      }
      else
      {
         g_months_done++;
         Print("exported ", year, "-", month, ": ", month_ticks, " ticks (total ", g_total_ticks, ")");
      }
      WriteStatus("RUNNING", IntegerToString(year) + "-" + IntegerToString(month));
      cursor = next_month;
   }

   WriteStatus("DONE", "total " + IntegerToString(g_total_ticks));
   Print("=== DONE: ", g_total_ticks, " real ticks, ", g_months_done, " months, ",
         g_months_empty, " empty months ===");
   Alert("AI Quant Terminal: exported ", g_total_ticks, " real ", InpSymbol, " ticks");
}
