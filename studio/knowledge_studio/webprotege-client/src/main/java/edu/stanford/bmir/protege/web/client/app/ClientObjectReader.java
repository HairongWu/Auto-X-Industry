package edu.stanford.bmir.protege.web.client.app;

import com.google.gwt.core.client.JavaScriptObject;
import com.google.gwt.json.client.JSONObject;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 23/12/14
 */
public class ClientObjectReader<T> {

    private final String variableName;

    private final ClientObjectDecoder<T> decoder;

    public ClientObjectReader(String variableName, ClientObjectDecoder<T> decoder) {
        this.variableName = variableName;
        this.decoder = decoder;
    }

    public static <T> ClientObjectReader<T> create(String variableName, ClientObjectDecoder<T> decoder) {
        return new ClientObjectReader<T>(variableName, decoder);
    }

    public T read() {
        JavaScriptObject value = readVariable(variableName);
        return decoder.decode(new JSONObject(value));
    }


    private native JavaScriptObject readVariable(String name)/*-{
        return $wnd[name];
    }-*/;
}
