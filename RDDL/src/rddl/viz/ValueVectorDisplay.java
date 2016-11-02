package rddl.viz;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Map;

import rddl.EvalException;
import rddl.State;
import rddl.RDDL.LCONST;
import rddl.RDDL.PVARIABLE_DEF;
import rddl.RDDL.PVARIABLE_INTERM_DEF;
import rddl.RDDL.PVAR_NAME;

public class ValueVectorDisplay extends StateViz {

	public boolean _bSuppressNonFluents = false;
	public boolean _bSuppressIntermFluents = false;
	public boolean _bSuppressActionFluents = false;
	public boolean _bSuppressWriteFile = false;
	public String _sDataPath = null;
	public String _sLabelPath = null;
	public PrintWriter _pDataOut = null;
	public PrintWriter _pLabelOut = null;
	
	public ValueVectorDisplay() {
		_bSuppressNonFluents = true;
	}

	public ValueVectorDisplay(boolean suppress_nonfluents,boolean suppress_intermfluents) {
		_bSuppressNonFluents = suppress_nonfluents;
		_bSuppressIntermFluents = suppress_intermfluents;
	}
	
	public void StateOnly(){
		_bSuppressActionFluents = true;
		_bSuppressNonFluents = true;
		_bSuppressIntermFluents = true;		
	}
	
	public void StateAction(){
		_bSuppressActionFluents = false;
		_bSuppressNonFluents = true;
		_bSuppressIntermFluents = true;
	}
	
	public void WriteFile(String data_path, String label_path){
		_bSuppressWriteFile = true;
		_sDataPath = data_path;
		_sLabelPath = label_path;		
		_pDataOut = getFileHandler(data_path);
		_pLabelOut = getFileHandler(label_path);
	}
	
	public void close(){
		if(_pDataOut!=null&&_pLabelOut!=null){
			_pDataOut.close();
			_pLabelOut.close();
		}
	}
	
	private PrintWriter getFileHandler(String path){
		FileWriter fw = null;
		BufferedWriter bw = null;
		PrintWriter out = null;
		try {
		    fw = new FileWriter(path, true);
		    bw = new BufferedWriter(fw);
		    out = new PrintWriter(bw);
		} catch (IOException e) {
		    System.out.println("There is no existing directory for file:"+path);
		}		
		return out;
	}
	

	@Override
	public void display(State s, int time) {
		// TODO Auto-generated method stub
		String vector = getStateDescription(s);
		System.out.println(vector);
		if(_bSuppressWriteFile == true){
			if(_bSuppressActionFluents == true){
				_pLabelOut.println(vector);
			}else{
				_pDataOut.println(vector);
			}
		}
	}
	
	public String getStateDescription(State s) {
		StringBuilder sb = new StringBuilder("");
		
		// Go through all variable types (state, interm, observ, action, nonfluent)
		for (Map.Entry<String,ArrayList<PVAR_NAME>> e : s._hmTypeMap.entrySet()) {
			
			if (_bSuppressNonFluents && e.getKey().equals("nonfluent"))
				continue;
			
			if (_bSuppressIntermFluents && e.getKey().equals("interm"))
				continue;
			
			if (_bSuppressActionFluents && e.getKey().equals("action"))
				continue;
			
			// Go through all variable names p for a variable type
			for (PVAR_NAME p : e.getValue()) {

				String var_type = e.getKey();
				try {
					// Go through all term groundings for variable p
					ArrayList<ArrayList<LCONST>> gfluents = s.generateAtoms(p);										
					for (int i=0; i<gfluents.size();i++)
						sb.append(s.getPVariableAssign(p, gfluents.get(i))+((i+1)==gfluents.size()? "" : ","));
					sb.append(",");
						
				} catch (EvalException ex) {
					sb.append("- could not retrieve assignment " + s + " for " + p + "\n");
				}
			}
		}
		String output = sb.toString();		
		return output.substring(0, output.length()-1);
	}

}
