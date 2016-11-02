package util;

public class ArgsParser {
	
	public static int getOptionPos(String flag, String[] args) {
		if (args == null)
			return -1;
			
		for (int i = 0; i < args.length; i++) {
			if ((args[i].length() > 0) && (args[i].charAt(0) == '-')) {
			// Check if it is a negative number
				try {
					Double.valueOf(args[i]);
				} 
				catch (NumberFormatException e) {
					// found?
					if (args[i].equals("-" + flag))
						return i;
				}
			  }
			}
		return -1;
	}
	
	  public static String getOption(String flag, String[] options) throws Exception {		
		  String newString;
		  int i = getOptionPos(flag, options);
		
		  if (i > -1) {
			  if (options[i].equals("-" + flag)) {
				  if (i + 1 == options.length) {
					  throw new Exception("No value given for -" + flag + " option.");
				  }
				  options[i] = "";
				  newString = new String(options[i + 1]);
				  options[i + 1] = "";
				  return newString;
			  }
			  if (options[i].charAt(1) == '-') {
				  return "";
			  }
		  }		
		  return "";
	  }	
}
