package edu.stanford.bmir.protege.web.server.persistence;

import com.mongodb.MongoClient;
import edu.stanford.bmir.protege.web.server.app.ApplicationDisposablesManager;
import edu.stanford.bmir.protege.web.server.collection.CollectionIdConverter;
import edu.stanford.bmir.protege.web.server.color.ColorConverter;
import edu.stanford.bmir.protege.web.server.form.FormIdConverter;
import edu.stanford.bmir.protege.web.server.inject.MongoClientProvider;
import edu.stanford.bmir.protege.web.server.tag.TagIdConverter;
import edu.stanford.bmir.protege.web.server.util.DisposableObjectManager;
import org.mongodb.morphia.Morphia;
import uk.ac.manchester.cs.owl.owlapi.OWLDataFactoryImpl;

import java.util.Optional;

/**
 * Matthew Horridge
 * Stanford Center for Biomedical Informatics Research
 * 6 Oct 2016
 */
public class MongoTestUtils {

    private static final String TEST_DB_NAME = "webprotege-test";

    public static MongoClient createMongoClient() {
        return new MongoClientProvider("localhost", 27017, Optional.empty(), new ApplicationDisposablesManager(new DisposableObjectManager())).get();
    }

    public static Morphia createMorphia() {
        return new MorphiaProvider(
                new UserIdConverter(),
                new OWLEntityConverter(new OWLDataFactoryImpl()),
                new ProjectIdConverter(),
                new ThreadIdConverter(),
                new CommentIdConverter(),
                new CollectionIdConverter(),
                new FormIdConverter(),
                new TagIdConverter(),
                new ColorConverter()).get();
    }


    public static String getTestDbName() {
        return TEST_DB_NAME;
    }
}
