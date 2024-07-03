package edu.stanford.bmir.protege.web.server.cmdline;

import com.mongodb.MongoClient;
import com.mongodb.ServerAddress;
import edu.stanford.bmir.protege.web.server.access.AccessManager;
import edu.stanford.bmir.protege.web.server.access.AccessManagerImpl;
import edu.stanford.bmir.protege.web.server.access.RoleOracleImpl;
import edu.stanford.bmir.protege.web.server.collection.CollectionIdConverter;
import edu.stanford.bmir.protege.web.server.form.FormIdConverter;
import edu.stanford.bmir.protege.web.server.persistence.*;
import edu.stanford.bmir.protege.web.server.color.ColorConverter;
import edu.stanford.bmir.protege.web.server.tag.TagIdConverter;
import org.mongodb.morphia.Datastore;
import org.mongodb.morphia.Morphia;
import uk.ac.manchester.cs.owl.owlapi.OWLDataFactoryImpl;

import java.io.Console;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 11 Apr 2017
 */
public class RebuildPermissions {

    public void run() {
        System.out.println("Rebuilding permissions...");
        MongoClient mongoClient = WebProtegeCli.getMongoClient();
        MorphiaProvider morphiaProvider = new MorphiaProvider(
                new UserIdConverter(),
                new OWLEntityConverter(new OWLDataFactoryImpl()),
                new ProjectIdConverter(),
                new ThreadIdConverter(),
                new CommentIdConverter(),
                new CollectionIdConverter(), new FormIdConverter(), new TagIdConverter(), new ColorConverter());
        Morphia morphia = morphiaProvider.get();
        Datastore datastore = morphia.createDatastore(mongoClient, "webprotege");
        AccessManager accessManager = new AccessManagerImpl(
                RoleOracleImpl.get(),
                datastore
        );
        accessManager.rebuild();
        mongoClient.close();
        System.out.println("Finished rebuilding permissions");
    }

    public static void main(String[] args) {
        RebuildPermissions rebuildPermissions = new RebuildPermissions();
        rebuildPermissions.run();
    }
}
