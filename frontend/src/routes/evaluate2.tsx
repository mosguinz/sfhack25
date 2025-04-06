import {
  Box,
  Button,
  Grid,
  Heading,
  Image,
  Input,
  SimpleGrid,
  Skeleton,
  Text,
  VStack,
} from "@chakra-ui/react"
import { createFileRoute } from "@tanstack/react-router"
import { useState } from "react"

type EvaluationResults = {
  diagnosis: string
  confidence: number
  // recommendations: string
}

export const Route = createFileRoute("/evaluate2")({
  component: ImageEvaluation,
  beforeLoad: async () => {
    // TODO: handle unauthorized here
  },
})

function ImageEvaluation() {
  const [image, setImage] = useState<string | null>(null)
  const [storeImages, setStoreImages] = useState<File>();
  const [isEvaluating, setIsEvaluating] = useState(false)
  const [results, setResults] = useState<EvaluationResults | null>(null)
  // const [results, setResults] = useState<EvaluationResults[]>([]);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {

      setStoreImages(file)

      setResults(null)
      const reader = new FileReader()
      reader.onloadend = () => {
        setImage(reader.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const removeImage = () => {
    setImage(null)
    setResults(null)
  }

  const handleEvaluate = async () => {
    setIsEvaluating(true)
    setResults(null)
    setTimeout(() => {
      setResults({
        diagnosis: "Normal",
        confidence: 0.95,
        // recommendations: "Regular follow-up recommended",
      })
      setIsEvaluating(false)
    }, 2500)
  }

  return (
    <Box p={8}>
      <Grid templateColumns={{ base: "1fr", lg: "2fr 1fr" }} gap={8}>
        {/* Main Upload Section */}
        <Box>
          <Heading size="3xl" mb={4}>
            Chest X-ray Analyzer
          </Heading>
          <Box h="1px" bg="gray.200" mb={4} />
          <Box mt={2}>
            <ImageUploadBox
              label="Chest X-ray"
              image={image}
              onUpload={handleImageUpload}
              onRemove={removeImage}
              mt={4}
              h="400px"
              isAnalyzing={isEvaluating}
            />
          </Box>
        </Box>

        {/* Right Sidebar */}
        <VStack align="stretch" gap={6} mt={16}>
          <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
            <Heading 
              size="md" 
              mb={2}
              color={image ? "inherit" : "gray.400"}
              transition="color 0.2s ease-in-out"
            >
              Ready to analyze
            </Heading>
            <Button
              colorScheme="brand"
              onClick={handleEvaluate}
              loading={isEvaluating}
              disabled={!image}
              w="full"
            >
              Analyze
            </Button>
          </Box>

          <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
            <Heading size="md" mb={2}>
              Patient Information
            </Heading>
            <Text color="gray.500">[Form or metadata goes here]</Text>
          </Box>

          {results && image && (
            <Box p={4} borderWidth="1px" rounded="xl" bg="white" boxShadow="sm">
              <Heading size="md" mb={2}>
                Analysis Results
              </Heading>
              <Text>
                <strong>Diagnosis:</strong> {results.diagnosis}
              </Text>
              <Text>
                <strong>Confidence:</strong>{" "}
                {(results.confidence * 100).toFixed(2)}%
              </Text>
              <Text>
                {/* <strong>Recommendations:</strong> {results.recommendations} */}
              </Text>
            </Box>
          )}
        </VStack>
      </Grid>
    </Box>
  )
}

function ImageUploadBox({
  label,
  image,
  onUpload,
  onRemove,
  mt,
  h,
  isAnalyzing,
}: {
  label: string
  image: string | null
  onUpload: (e: React.ChangeEvent<HTMLInputElement>) => void
  onRemove: () => void
  mt?: number
  h?: string
  isAnalyzing?: boolean
}) {
  const [isLoading, setIsLoading] = useState(false)
  const [isImageLoaded, setIsImageLoaded] = useState(false)

  const handleUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    setIsLoading(true)
    setIsImageLoaded(false)
    onUpload(e)
  }

  const handleImageLoad = () => {
    setIsImageLoaded(true)
    setIsLoading(false)
  }

  return (
    <Box>
      <style>
        {`
          @keyframes scanVertical {
            from { top: 0; }
            to { top: 100%; }
          }
          @keyframes scanHorizontal {
            from { left: 0; }
            to { left: 100%; }
          }
        `}
      </style>
      {image ? (
        <Box
          position="relative"
          rounded="xl"
          overflow="hidden"
          transition="all 0.2s"
          boxShadow="0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)"
          border="1px solid"
          borderColor={isAnalyzing ? "black" : "transparent"}
          _hover={{
            border: "1px solid",
            borderColor: "brand.300"
          }}
        >
          <Text 
            position="absolute"
            top={2}
            left={2}
            bg="whiteAlpha.800"
            px={2}
            py={1}
            rounded="md"
            fontSize="sm"
            fontWeight="medium"
            color="gray.700"
            zIndex={1}
          >
            {label}
          </Text>
          <Box position="relative" h={h || "180px"} display="flex" alignItems="center" justifyContent="center">
            <Image
              src={image}
              alt={label}
              w="full"
              h={h}
              objectFit="contain"
              bg="gray.50"
              onLoad={handleImageLoad}
              opacity={isImageLoaded ? 1 : 0}
              transition="opacity 0.2s ease-in-out"
            />
            {!isImageLoaded && (
              <Skeleton
                position="absolute"
                top={0}
                left={0}
                right={0}
                bottom={0}
              />
            )}
            {isAnalyzing && (
              <>
                <Box
                  position="absolute"
                  left={0}
                  right={0}
                  height="2px"
                  bg="red.500"
                  boxShadow="0 0 10px rgba(255, 0, 0, 0.5)"
                  style={{
                    animation: "scanVertical 1.25s linear infinite alternate"
                  }}
                />
                <Box
                  position="absolute"
                  top={0}
                  bottom={0}
                  width="2px"
                  bg="red.500"
                  boxShadow="0 0 10px rgba(255, 0, 0, 0.5)"
                  style={{
                    animation: "scanHorizontal 1.25s linear infinite alternate"
                  }}
                />
              </>
            )}
          </Box>
          <Button
            position="absolute"
            top={2}
            right={2}
            size="sm"
            colorScheme="red"
            onClick={onRemove}
            zIndex={1}
          >
            Remove
          </Button>
        </Box>
      ) : (
        <label
          htmlFor="image-upload"
          style={{
            cursor: "pointer",
            borderWidth: "2px",
            borderStyle: "dashed",
            borderColor: "gray.200",
            borderRadius: "0.75rem",
            padding: "1.5rem",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            transition: "all 0.2s",
            height: "400px",
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.borderColor = "var(--chakra-colors-brand-300)";
            e.currentTarget.style.backgroundColor = "var(--chakra-colors-gray-50)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.borderColor = "var(--chakra-colors-gray-200)";
            e.currentTarget.style.backgroundColor = "transparent";
          }}
        >
          <Text fontSize="4xl" color="gray.400">
            +
          </Text>
          <Text color="gray.500" fontSize="sm" mt={2}>
            Click to upload chest X-ray image
          </Text>
          <Input
            type="file"
            id="image-upload"
            accept="image/*,.dcm"
            onChange={handleUpload}
            display="none"
          />
        </label>
      )}
    </Box>
  )
}
