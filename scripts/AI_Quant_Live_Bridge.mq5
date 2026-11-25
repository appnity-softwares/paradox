//+------------------------------------------------------------------+
//|                                       AI_Quant_Live_Bridge.mq5   |
//|               Two-Way Tick & Execution Bridge for AI Quant Terminal|
//|                                    http://127.0.0.1:8123         |
//+------------------------------------------------------------------+
#property copyright "AI Quant Terminal"
#property link      "http://127.0.0.1:8123"
#property version   "2.00"

#include <Trade\Trade.mqh>

input string InpTickUrl    = "http://127.0.0.1:8123/api/mt5/tick";     // Tick Webhook
input string InpCmdUrl     = "http://127.0.0.1:8123/api/mt5/commands"; // Command Polling URL
input ulong  InpMagic      = 20261117;                                 // Magic Number
input int    InpPollMs     = 250;                                      // Command Poll Interval (ms)

CTrade g_trade;

int OnInit()
{
   g_trade.SetExpertMagicNumber(InpMagic);
   g_trade.SetDeviationInPoints(30);
   g_trade.SetTypeFilling(ORDER_FILLING_IOC);

   Print("=================================================");
   Print("⚡ AI QUANT TERMINAL TWO-WAY BRIDGE ACTIVE");
   Print("Symbol: ", _Symbol, " | Magic: ", InpMagic);
   Print("Broadcasting live ticks -> ", InpTickUrl);
   Print("Listening for trade commands -> ", InpCmdUrl);
   Print("=================================================");

   EventSetMillisecondTimer(InpPollMs);
   return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason)
{
   EventKillTimer();
   Print("AI Quant Terminal Bridge Stopped.");
}

ulong g_last_tick_sent_ms = 0;

//--- Send real-time tick to dashboard
void SendTickNow()
{
   MqlTick tick;
   if(SymbolInfoTick(_Symbol, tick))
   {
      double p_last = tick.last > 0 ? tick.last : ((tick.bid + tick.ask) / 2.0);
      long t_ms = (long)tick.time_msc;
      if(t_ms <= 0) t_ms = (long)TimeCurrent() * 1000;

      string json = StringFormat(
         "{\"symbol\":\"%s\",\"bid\":%.2f,\"ask\":%.2f,\"last\":%.2f,\"volume\":%d,\"time_ms\":%I64d}",
         _Symbol, tick.bid, tick.ask, p_last, (tick.volume > 0 ? tick.volume : 1), t_ms
      );
      char postData[];
      StringToCharArray(json, postData, 0, WHOLE_ARRAY, CP_UTF8);
      char result[];
      string resultHeaders;
      int res = WebRequest("POST", InpTickUrl, "Content-Type: application/json\r\n", 100, postData, result, resultHeaders);
      if(res == 200 || res == 201)
      {
         g_last_tick_sent_ms = GetTickCount64();
      }
      else if(res == -1)
      {
         int err = GetLastError();
         if(err == 4060)
         {
            Print("⚠️ WebRequest blocked (error 4060). In MT5: Tools -> Options -> Expert Advisors -> check 'Allow WebRequest for listed URL' and add 'http://127.0.0.1:8123'");
         }
      }
   }
}

//--- Broadcast real-time tick to dashboard
void OnTick()
{
   SendTickNow();
}

//--- Poll dashboard for execution commands (BUY, SELL, CLOSE, MODIFY) and keep tick feed live
void OnTimer()
{
   // 1. If no tick sent in last 1000ms, broadcast tick heartbeat
   if(GetTickCount64() - g_last_tick_sent_ms >= 1000)
   {
      SendTickNow();
   }

   // 2. Poll for trade commands
   char postData[];
   char result[];
   string resultHeaders;
   int res = WebRequest("GET", InpCmdUrl, "", 100, postData, result, resultHeaders);
   if(res == 200 && ArraySize(result) > 2)
   {
      string resp = CharArrayToString(result, 0, WHOLE_ARRAY, CP_UTF8);
      if(StringFind(resp, "\"action\"") >= 0)
      {
         ProcessCommand(resp);
      }
   }
}

//--- Execute trade command locally inside MT5
void ProcessCommand(string json)
{
   Print(">>> EXECUTING TERMINAL COMMAND: ", json);

   // Simple fast JSON value extractor
   string action = ExtractJsonValue(json, "action");
   string side   = ExtractJsonValue(json, "side");
   string sym    = ExtractJsonValue(json, "symbol");
   double vol    = StringToDouble(ExtractJsonValue(json, "volume"));
   double sl     = StringToDouble(ExtractJsonValue(json, "sl"));
   double tp     = StringToDouble(ExtractJsonValue(json, "tp"));
   ulong  ticket = (ulong)StringToInteger(ExtractJsonValue(json, "ticket"));

   if(sym == "") sym = _Symbol;
   if(vol <= 0.0) vol = 0.01;

   if(action == "OPEN")
   {
      if(side == "BUY")
      {
         Print("Bridge: Placing BUY ", vol, " ", sym, " SL=", sl, " TP=", tp);
         g_trade.Buy(vol, sym, 0, sl, tp, "AI-Quant Buy");
      }
      else if(side == "SELL")
      {
         Print("Bridge: Placing SELL ", vol, " ", sym, " SL=", sl, " TP=", tp);
         g_trade.Sell(vol, sym, 0, sl, tp, "AI-Quant Sell");
      }
   }
   else if(action == "CLOSE")
   {
      Print("Bridge: Closing position ticket ", ticket);
      g_trade.PositionClose(ticket);
   }
   else if(action == "MODIFY")
   {
      Print("Bridge: Modifying SL/TP for ticket ", ticket, " SL=", sl, " TP=", tp);
      g_trade.PositionModify(ticket, sl, tp);
   }
}

string ExtractJsonValue(string json, string key)
{
   string pattern = "\"" + key + "\":\"";
   int start = StringFind(json, pattern);
   if(start >= 0)
   {
      start += StringLen(pattern);
      int end = StringFind(json, "\"", start);
      if(end > start) return StringSubstr(json, start, end - start);
   }
   // try numeric
   pattern = "\"" + key + "\":";
   start = StringFind(json, pattern);
   if(start >= 0)
   {
      start += StringLen(pattern);
      int end = StringFind(json, ",", start);
      if(end < 0) end = StringFind(json, "}", start);
      if(end > start)
      {
         string val = StringSubstr(json, start, end - start);
         StringTrimLeft(val);
         StringTrimRight(val);
         return val;
      }
   }
   return "";
}
